import queue
import threading
from dataclasses import dataclass
from typing import Protocol

from ..config.training import BeetleConfig
from .audio import AudioPreprocessor
from .collate import BatchCollator, Tokenizer
from .index import DatabaseSegmentIndex
from .loader import DatabaseBatchLoader
from .records import BeetleBatch, PlannedBatch
from .sampling import (
    ContinuousBatchPlanner,
    DistributedShard,
    PlannedWindowBatch,
    PlannerState,
)

TrainingBatch = BeetleBatch


class PrefetchCallbacks(Protocol):
    def check_cancel(self) -> None: ...


@dataclass(frozen=True)
class DataPipelineState:
    data_fingerprint: str
    planner: PlannerState
    world_size: int

@dataclass(frozen=True)
class _QueuedBatch:
    batch: TrainingBatch
    state_after: PlannerState
    decoded_bytes: int


@dataclass(frozen=True)
class _QueuedWindow:
    batches: tuple[_QueuedBatch, ...]


@dataclass(frozen=True)
class _ProducerFailure:
    error: BaseException


class TrainingDataPipeline:
    def __init__(
        self,
        planner: ContinuousBatchPlanner,
        loader: DatabaseBatchLoader,
        callbacks: PrefetchCallbacks,
        window_size: int,
        maximum_decoded_bytes: int,
        sample_rate: int,
        initial_state: DataPipelineState,
    ) -> None:
        planner.load_state_dict(initial_state.planner)
        self.planner = planner
        self.loader = loader
        self.callbacks = callbacks
        self.window_size = window_size
        self.maximum_decoded_bytes = maximum_decoded_bytes
        self.sample_rate = sample_rate
        self._committed_state = initial_state
        self._windows: queue.Queue[_QueuedWindow | _ProducerFailure] = queue.Queue(1)
        self._condition = threading.Condition()
        self._reserved_bytes = 0
        self._active: list[_QueuedBatch] = []
        self._in_flight: _QueuedBatch | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start()

    def next_batch(self) -> TrainingBatch:
        if self._in_flight is not None:
            raise RuntimeError("mark the current batch consumed before requesting another")
        if not self._active:
            self._active.extend(self._next_window().batches)
        self._in_flight = self._active.pop(0)
        return self._in_flight.batch

    def mark_consumed(self) -> None:
        if self._in_flight is None:
            raise RuntimeError("no fetched batch is awaiting consumption")
        self._committed_state = DataPipelineState(
            self.planner.index.fingerprint,
            self._in_flight.state_after,
            self.planner.shard.world_size,
        )
        self._release_bytes(self._in_flight.decoded_bytes)
        self._in_flight = None

    def state_dict(self) -> DataPipelineState:
        return self._committed_state

    def load_state_dict(self, state: DataPipelineState) -> None:
        if self._in_flight is not None:
            raise RuntimeError("cannot restore while a batch is in flight")
        self._stop_producer()
        self.planner.load_state_dict(state.planner)
        self._committed_state = state
        self._windows = queue.Queue(1)
        self._active = []
        self._reserved_bytes = 0
        self._stop = threading.Event()
        self._start()

    def close(self) -> None:
        self._stop_producer()

    def _next_window(self) -> _QueuedWindow:
        while True:
            self.callbacks.check_cancel()
            try:
                item = self._windows.get(timeout=0.1)
                break
            except queue.Empty:
                if self._thread is not None and not self._thread.is_alive():
                    raise RuntimeError("prefetch producer stopped without a result")
        if isinstance(item, _ProducerFailure):
            raise item.error
        return item

    def _stop_producer(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("prefetch producer did not stop")
            self._thread = None

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._produce,
            name="beetle-window-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        reserved = 0
        try:
            while not self._stop.is_set():
                plans = self.planner.next_window(self.window_size)
                estimates = tuple(self._estimate(item.batch) for item in plans)
                reserved = sum(estimates)
                if not self._reserve_bytes(reserved):
                    return
                queued = tuple(
                    self._prepare_batch(item, decoded_bytes)
                    for item, decoded_bytes in zip(plans, estimates, strict=True)
                )
                if not self._put(_QueuedWindow(queued)):
                    self._release_bytes(reserved)
                    return
                reserved = 0
        except BaseException as error:
            if reserved:
                self._release_bytes(reserved)
            self._put(_ProducerFailure(error))

    def _prepare_batch(
        self,
        planned: PlannedWindowBatch,
        decoded_bytes: int,
    ) -> _QueuedBatch:
        return _QueuedBatch(
            self.loader.load(planned.batch),
            planned.state_after,
            decoded_bytes,
        )

    def _estimate(self, planned: PlannedBatch) -> int:
        ranges = set()
        for example in planned.examples:
            ranges.add((example.key.audio_file_id, example.target.start, example.target.end))
        for group in (*planned.voice_groups, *planned.style_groups):
            for view in group.views:
                ranges.add((view.key.audio_file_id, view.audio.start, view.audio.end))
        seconds = sum(end - start for _, start, end in ranges)
        return max(1, round(seconds * self.sample_rate * 4))

    def _reserve_bytes(self, decoded_bytes: int) -> bool:
        with self._condition:
            self._condition.wait_for(
                lambda: self._stop.is_set()
                or self._reserved_bytes + decoded_bytes
                <= self.maximum_decoded_bytes
            )
            if self._stop.is_set():
                return False
            self._reserved_bytes += decoded_bytes
            return True

    def _release_bytes(self, decoded_bytes: int) -> None:
        with self._condition:
            self._reserved_bytes -= decoded_bytes
            self._condition.notify_all()

    def _put(self, item: _QueuedWindow | _ProducerFailure) -> bool:
        while not self._stop.is_set():
            try:
                self._windows.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False


def build_data_pipeline(
    config: BeetleConfig,
    callbacks: PrefetchCallbacks,
    index: DatabaseSegmentIndex,
    phoneme_tokenizer: Tokenizer,
    text_tokenizer: Tokenizer,
    initial_state: DataPipelineState,
    shard: DistributedShard,
) -> TrainingDataPipeline:
    audio = config.audio
    preprocessor = AudioPreprocessor(
        audio.sample_rate,
        audio.n_fft,
        audio.win_length,
        audio.hop_length,
        audio.mel_channels,
        audio.f_min,
        audio.f_max,
    )
    planner = ContinuousBatchPlanner(
        index=index,
        batch_size=config.training.batch_size,
        seed=config.runtime.seed,
        maximum_seconds=config.data.maximum_seconds,
        grouping=config.data.grouping,
        shard=shard,
    )
    collator = BatchCollator(
        preprocessor,
        phoneme_tokenizer,
        text_tokenizer,
        config.data.augmentation,
        config.architecture.language.values,
        config.adversarial.segment_samples // config.audio.hop_length,
    )
    loader = DatabaseBatchLoader(index, collator)
    return TrainingDataPipeline(
        planner=planner,
        loader=loader,
        callbacks=callbacks,
        window_size=config.data.prefetch.window_size,
        maximum_decoded_bytes=config.data.prefetch.decoded_bytes,
        sample_rate=audio.sample_rate,
        initial_state=initial_state,
    )
