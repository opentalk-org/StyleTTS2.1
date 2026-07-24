import queue
import threading
from dataclasses import dataclass
from typing import Protocol

from .records import BeetleBatch, PlannedBatch
from .sampling import ContinuousBatchPlanner, PlannedWindowBatch, PlannerState
from .source import FetchedBatch

TrainingBatch = BeetleBatch


class PrefetchCallbacks(Protocol):
    def check_cancel(self) -> None: ...


class PlannedBatchLoader(Protocol):
    def fetch(self, planned: PlannedBatch) -> FetchedBatch: ...

    def collate(self, fetched: FetchedBatch) -> TrainingBatch: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DataPipelineState:
    data_fingerprint: str
    planner: PlannerState
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("pipeline world size must be positive")


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


class BoundedBatchPrefetcher:
    def __init__(
        self,
        planner: ContinuousBatchPlanner,
        loader: PlannedBatchLoader,
        callbacks: PrefetchCallbacks,
        window_size: int,
        maximum_decoded_bytes: int,
        sample_rate: int,
        initial_state: DataPipelineState,
    ) -> None:
        if window_size <= 0:
            raise ValueError("prefetch window size must be positive")
        if initial_state.data_fingerprint != planner.index.fingerprint:
            raise ValueError("data fingerprint does not match planner index")
        if initial_state.world_size != planner.shard.world_size:
            raise ValueError("pipeline world size does not match planner shard")
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

    @property
    def queued_bytes(self) -> int:
        with self._condition:
            return self._reserved_bytes

    def next_batch(self) -> TrainingBatch:
        if self._in_flight is not None:
            raise RuntimeError(
                "mark the current batch consumed before requesting another"
            )
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
        if state.data_fingerprint != self.planner.index.fingerprint:
            raise ValueError("restored data fingerprint does not match index")
        if state.world_size != self.planner.shard.world_size:
            raise ValueError("restored world size does not match planner shard")
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
        self.loader.close()

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
        fetched = self.loader.fetch(planned.batch)
        return _QueuedBatch(
            self.loader.collate(fetched),
            planned.state_after,
            decoded_bytes,
        )

    def _estimate(self, planned: PlannedBatch) -> int:
        ranges = set()
        for example in planned.examples:
            ranges.add(
                (example.key.audio_file_id, example.target.start, example.target.end)
            )
        for group in (*planned.voice_groups, *planned.style_groups):
            for view in group.views:
                ranges.add((view.key.audio_file_id, view.audio.start, view.audio.end))
        seconds = sum(end - start for _, start, end in ranges)
        return max(1, round(seconds * self.sample_rate * 4))

    def _reserve_bytes(self, decoded_bytes: int) -> bool:
        if decoded_bytes > self.maximum_decoded_bytes:
            raise ValueError(
                "one prefetch window exceeds decoded-byte limit: "
                f"{decoded_bytes} > {self.maximum_decoded_bytes}"
            )
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._stop.is_set()
                    or self._reserved_bytes + decoded_bytes
                    <= self.maximum_decoded_bytes
                )
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
