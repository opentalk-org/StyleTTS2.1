import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch.utils.data import DataLoader

from ..profiling import profiling_fn

Batch = TypeVar("Batch")


@dataclass(frozen=True)
class _LoadedBatch(Generic[Batch]):
    value: Batch


@dataclass(frozen=True)
class _LoadFailure:
    error: BaseException


@dataclass(frozen=True)
class _EndOfEpoch:
    pass


_QueueItem = _LoadedBatch[Batch] | _LoadFailure | _EndOfEpoch


class PrefetchedDataPipeline(Generic[Batch]):
    """Overlap database reads, audio decoding, and collation with GPU training."""

    def __init__(self, loader: DataLoader, queued_batches: int = 2) -> None:
        self.loader = loader
        self.queued_batches = queued_batches

    def __len__(self) -> int:
        return len(self.loader)

    def prepare(self, accelerator) -> "PrefetchedDataPipeline[Batch]":
        return PrefetchedDataPipeline(
            accelerator.prepare(self.loader),
            self.queued_batches,
        )

    def __iter__(self) -> Iterator[Batch]:
        items: queue.Queue[_QueueItem] = queue.Queue(self.queued_batches)
        stopped = threading.Event()
        producer = threading.Thread(
            target=self._produce,
            args=(items, stopped),
            name="batch-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                with profiling_fn("data_collection"):
                    item = items.get()
                if isinstance(item, _LoadedBatch):
                    yield item.value
                elif isinstance(item, _LoadFailure):
                    raise item.error
                else:
                    break
        finally:
            stopped.set()
            producer.join(timeout=10)
            if producer.is_alive():
                raise RuntimeError("data prefetch producer did not stop")

    def _produce(
        self,
        items: queue.Queue[_QueueItem],
        stopped: threading.Event,
    ) -> None:
        try:
            for batch in self.loader:
                if not self._put(items, _LoadedBatch(batch), stopped):
                    return
            self._put(items, _EndOfEpoch(), stopped)
        except BaseException as error:
            self._put(items, _LoadFailure(error), stopped)

    @staticmethod
    def _put(
        items: queue.Queue[_QueueItem],
        item: _QueueItem,
        stopped: threading.Event,
    ) -> bool:
        while not stopped.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False
