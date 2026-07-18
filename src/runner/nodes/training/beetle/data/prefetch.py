import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .records import BeetleBatch, PlannedBatch
from .sampling import ContinuousBatchPlanner, PlannerState
from .source import FetchedBatch


class PrefetchCallbacks(Protocol):
    def check_cancel(self) -> None: ...


class PlannedBatchLoader(Protocol):
    def fetch(self, planned: PlannedBatch) -> FetchedBatch: ...

    def collate(self, fetched: FetchedBatch) -> BeetleBatch: ...

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
    batch: BeetleBatch
    state_after: PlannerState
    decoded_bytes: int


@dataclass(frozen=True)
class _PlannedCandidate:
    batch: PlannedBatch
    state_after: PlannerState
    decoded_bytes: int


@dataclass(frozen=True)
class _PendingFetch:
    candidate: _PlannedCandidate
    future: Future[FetchedBatch]


@dataclass(frozen=True)
class _ProducerFailure:
    error: BaseException


class BoundedBatchPrefetcher:
    def __init__(
        self,
        planner: ContinuousBatchPlanner,
        loader: PlannedBatchLoader,
        callbacks: PrefetchCallbacks,
        maximum_batches: int,
        maximum_decoded_bytes: int,
        sample_rate: int,
        initial_state: DataPipelineState,
    ) -> None:
        if initial_state.data_fingerprint != planner.index.fingerprint:
            raise ValueError("data fingerprint does not match planner index")
        if initial_state.world_size != planner.shard.world_size:
            raise ValueError("pipeline world size does not match planner shard")
        planner.load_state_dict(initial_state.planner)
        self.planner = planner
        self.loader = loader
        self.callbacks = callbacks
        self.maximum_batches = maximum_batches
        self.maximum_decoded_bytes = maximum_decoded_bytes
        self.sample_rate = sample_rate
        self._committed_state = initial_state
        self._queue: queue.Queue[_QueuedBatch | _ProducerFailure] = queue.Queue(maximum_batches)
        self._condition = threading.Condition()
        self._queued_bytes = 0
        self._in_flight: _QueuedBatch | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start()

    @property
    def queued_bytes(self) -> int:
        with self._condition:
            return self._queued_bytes

    def next_batch(self) -> BeetleBatch:
        if self._in_flight is not None:
            raise RuntimeError("mark the current batch consumed before requesting another")
        while True:
            self.callbacks.check_cancel()
            try:
                item = self._queue.get(timeout=0.1)
                break
            except queue.Empty:
                if self._thread is not None and not self._thread.is_alive():
                    raise RuntimeError("prefetch producer stopped without a result")
        if isinstance(item, _ProducerFailure):
            raise item.error
        self._release_bytes(item.decoded_bytes)
        self._in_flight = item
        return item.batch

    def mark_consumed(self) -> None:
        if self._in_flight is None:
            raise RuntimeError("no fetched batch is awaiting consumption")
        self._committed_state = DataPipelineState(
            self.planner.index.fingerprint,
            self._in_flight.state_after,
            self.planner.shard.world_size,
        )
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
        self._queue = queue.Queue(self.maximum_batches)
        self._queued_bytes = 0
        self._stop = threading.Event()
        self._start()

    def close(self) -> None:
        self._stop_producer()
        self.loader.close()

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
            name="beetle-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        candidate: _PlannedCandidate | None = None
        pending: _PendingFetch | None = None
        completed: _PendingFetch | None = None
        try:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="beetle-audio-prefetch",
            ) as executor:
                while not self._stop.is_set():
                    self.callbacks.check_cancel()
                    if pending is None:
                        candidate = candidate or self._plan_candidate()
                        if not self._reserve_bytes(candidate.decoded_bytes, True):
                            break
                        pending = _PendingFetch(
                            candidate,
                            executor.submit(self.loader.fetch, candidate.batch),
                        )
                        candidate = None
                    completed = pending
                    pending = None
                    fetched = completed.future.result()
                    candidate = self._plan_candidate()
                    if self._reserve_bytes(candidate.decoded_bytes, False):
                        pending = _PendingFetch(
                            candidate,
                            executor.submit(self.loader.fetch, candidate.batch),
                        )
                        candidate = None
                    batch = self.loader.collate(fetched)
                    queued = _QueuedBatch(
                        batch,
                        completed.candidate.state_after,
                        completed.candidate.decoded_bytes,
                    )
                    if not self._put(queued):
                        self._release_bytes(
                            completed.candidate.decoded_bytes
                        )
                        completed = None
                        break
                    completed = None
            if pending is not None:
                pending.future.cancel()
                self._release_bytes(pending.candidate.decoded_bytes)
        except BaseException as error:
            if pending is not None:
                pending.future.cancel()
                self._release_bytes(pending.candidate.decoded_bytes)
            if completed is not None:
                self._release_bytes(completed.candidate.decoded_bytes)
            self._put(_ProducerFailure(error))

    def _plan_candidate(self) -> _PlannedCandidate:
        planned = self.planner.next_batch()
        decoded_bytes = _estimated_decoded_bytes(planned, self.sample_rate)
        if decoded_bytes > self.maximum_decoded_bytes:
            raise ValueError(
                "one planned batch exceeds prefetch decoded-byte limit: "
                f"{decoded_bytes} > {self.maximum_decoded_bytes}"
            )
        return _PlannedCandidate(planned, self.planner.state_dict(), decoded_bytes)

    def _reserve_bytes(self, decoded_bytes: int, wait: bool) -> bool:
        with self._condition:
            if wait:
                self._condition.wait_for(
                    lambda: self._stop.is_set()
                    or self._queued_bytes + decoded_bytes
                    <= self.maximum_decoded_bytes
                )
            fits = self._queued_bytes + decoded_bytes <= self.maximum_decoded_bytes
            if not self._stop.is_set() and fits:
                self._queued_bytes += decoded_bytes
                return True
            return False

    def _release_bytes(self, decoded_bytes: int) -> None:
        with self._condition:
            self._queued_bytes -= decoded_bytes
            self._condition.notify_all()

    def _put(self, item: _QueuedBatch | _ProducerFailure) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False


def _estimated_decoded_bytes(planned: PlannedBatch, sample_rate: int) -> int:
    ranges = set()
    for example in planned.examples:
        ranges.add((example.key.audio_file_id, example.target.start, example.target.end))
        if example.pre_context is not None:
            context = example.pre_context
            ranges.add((context.key.audio_file_id, context.audio.start, context.audio.end))
        if example.post_context is not None:
            context = example.post_context
            ranges.add((context.key.audio_file_id, context.audio.start, context.audio.end))
    for group in (*planned.voice_groups, *planned.style_groups):
        for view in group.views:
            ranges.add((view.key.audio_file_id, view.audio.start, view.audio.end))
    seconds = sum(end - start for _, start, end in ranges)
    return max(1, round(seconds * sample_rate * 4))
