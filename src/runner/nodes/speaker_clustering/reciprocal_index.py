from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.candidates import CandidateMatrix


@dataclass(frozen=True)
class ReciprocalSearchResult:
    found: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray


class RowSortedCandidateIndex:
    def __init__(
        self, directory: Path, item_count: int, neighbors: int, mode: str
    ) -> None:
        self.item_count = item_count
        self.neighbors = neighbors
        shape = (item_count, neighbors)
        self.target_ids = np.memmap(
            directory / "target_ids.i64", dtype=np.int64, mode=mode, shape=shape
        )
        self.scores = np.memmap(
            directory / "scores.f32", dtype=np.float32, mode=mode, shape=shape
        )
        self.ranks = np.memmap(
            directory / "ranks.i32", dtype=np.int32, mode=mode, shape=shape
        )

    @classmethod
    def create(
        cls,
        directory: Path,
        candidates: CandidateMatrix,
        block_rows: int,
        check_cancel: Callable[[], None] | None = None,
    ) -> RowSortedCandidateIndex:
        if block_rows <= 0:
            raise ValueError(f"reciprocal index block_rows must be positive, got {block_rows}")
        directory.mkdir(parents=True, exist_ok=False)
        index = cls(directory, candidates.item_count, candidates.neighbors, mode="w+")
        invalid_id = np.iinfo(np.int64).max
        original_ranks = np.arange(1, candidates.neighbors + 1, dtype=np.int32)
        try:
            for start in range(0, candidates.item_count, block_rows):
                if check_cancel is not None:
                    check_cancel()
                stop = min(start + block_rows, candidates.item_count)
                target_ids = np.asarray(candidates.row_ids[start:stop])
                scores = np.asarray(candidates.scores[start:stop])
                sort_keys = np.where(target_ids >= 0, target_ids, invalid_id)
                order = np.argsort(sort_keys, axis=1, kind="stable")
                index.target_ids[start:stop] = np.take_along_axis(
                    sort_keys, order, axis=1
                )
                index.scores[start:stop] = np.take_along_axis(scores, order, axis=1)
                ranks = np.broadcast_to(original_ranks, target_ids.shape)
                index.ranks[start:stop] = np.take_along_axis(ranks, order, axis=1)
            index.flush()
            return index
        except BaseException:
            index.close()
            raise

    def search(
        self,
        source_ids: np.ndarray,
        target_ids: np.ndarray,
        check_cancel: Callable[[], None] | None = None,
    ) -> ReciprocalSearchResult:
        if source_ids.shape != target_ids.shape or source_ids.ndim != 2:
            raise ValueError("reciprocal search IDs must be aligned matrices")
        if check_cancel is not None:
            check_cancel()
        valid = (source_ids >= 0) & (source_ids < self.item_count)
        safe_sources = np.where(valid, source_ids, 0)
        low = np.zeros(source_ids.shape, dtype=np.int32)
        high = np.full(source_ids.shape, self.neighbors, dtype=np.int32)
        while np.any(low < high):
            active = low < high
            middle = (low + high) // 2
            safe_middle = np.minimum(middle, self.neighbors - 1)
            values = np.asarray(self.target_ids[safe_sources, safe_middle])
            move_right = active & (values < target_ids)
            low = np.where(move_right, middle + 1, low)
            high = np.where(active & ~move_right, middle, high)
        in_bounds = low < self.neighbors
        safe_positions = np.minimum(low, self.neighbors - 1)
        matched_ids = np.asarray(self.target_ids[safe_sources, safe_positions])
        found = valid & in_bounds & (matched_ids == target_ids)
        scores = np.full(source_ids.shape, -np.inf, dtype=np.float32)
        ranks = np.zeros(source_ids.shape, dtype=np.int32)
        scores[found] = self.scores[safe_sources[found], safe_positions[found]]
        ranks[found] = self.ranks[safe_sources[found], safe_positions[found]]
        return ReciprocalSearchResult(found=found, scores=scores, ranks=ranks)

    def flush(self) -> None:
        self.target_ids.flush()
        self.scores.flush()
        self.ranks.flush()

    def close(self) -> None:
        self.target_ids._mmap.close()
        self.scores._mmap.close()
        self.ranks._mmap.close()
