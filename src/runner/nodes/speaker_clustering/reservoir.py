from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeterministicVectorReservoir:
    """Retain a fixed-capacity, row-ID-stable sample without matrix reallocations."""

    capacity: int
    random_seed: int
    _vectors: np.ndarray | None = None
    _row_ids: np.ndarray = field(init=False)
    _priorities: np.ndarray = field(init=False)
    _heap: list[tuple[int, int, int]] = field(default_factory=list)
    _count: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(
                f"training sample maximum_rows must be positive, got {self.capacity}"
            )
        self._row_ids = np.empty(self.capacity, dtype=np.int64)
        self._priorities = np.empty(self.capacity, dtype=np.uint64)

    def add(self, row_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(row_ids, dtype=np.int64)
        values = np.asarray(vectors, dtype=np.float32)
        if not len(ids):
            return
        if self._vectors is None:
            self._vectors = np.empty(
                (self.capacity, values.shape[1]), dtype=np.float32
            )
        priorities = stable_priorities(ids, self.random_seed)
        for row_id, priority, vector in zip(ids, priorities, values, strict=True):
            self._add_one(int(row_id), int(priority), vector)

    def result(self) -> np.ndarray:
        if self._vectors is None or self._count == 0:
            raise ValueError(
                "cannot train FAISS index without accepted speaker embeddings"
            )
        order = np.lexsort(
            (self._row_ids[: self._count], self._priorities[: self._count])
        )
        return np.ascontiguousarray(self._vectors[: self._count][order])

    def _add_one(self, row_id: int, priority: int, vector: np.ndarray) -> None:
        if self._count < self.capacity:
            slot = self._count
            self._store(slot, row_id, priority, vector)
            heapq.heappush(self._heap, (-priority, -row_id, slot))
            self._count += 1
            return
        worst_priority, worst_row_id, slot = self._heap[0]
        if (priority, row_id) >= (-worst_priority, -worst_row_id):
            return
        self._store(slot, row_id, priority, vector)
        heapq.heapreplace(self._heap, (-priority, -row_id, slot))

    def _store(
        self, slot: int, row_id: int, priority: int, vector: np.ndarray
    ) -> None:
        assert self._vectors is not None, "reservoir storage is not initialized"
        self._vectors[slot] = vector
        self._row_ids[slot] = row_id
        self._priorities[slot] = priority


def stable_priorities(row_ids: np.ndarray, random_seed: int) -> np.ndarray:
    values = np.asarray(row_ids, dtype=np.uint64) + np.uint64(random_seed)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))
