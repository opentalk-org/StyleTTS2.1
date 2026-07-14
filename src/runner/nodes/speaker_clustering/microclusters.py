from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.edge_shards import EdgeBlock


class MicroclusterLabels:
    def __init__(self, directory: Path, item_count: int, mode: str) -> None:
        if item_count <= 0:
            raise ValueError("microcluster label count must be positive")
        self.item_count = item_count
        self.values = np.memmap(
            directory / "labels.i64",
            dtype=np.int64,
            mode=mode,
            shape=(item_count,),
        )

    @classmethod
    def create(cls, directory: Path, item_count: int) -> MicroclusterLabels:
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory, item_count, mode="w+")

    def flush(self) -> None:
        self.values.flush()


def mutual_best_labels(
    item_count: int,
    accepted: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
) -> np.ndarray:
    best_neighbors = np.full(item_count, -1, dtype=np.int64)
    best_scores = np.full(item_count, -np.inf, dtype=np.float32)
    _find_best_neighbors(best_neighbors, best_scores, edge_blocks, None)
    return _pair_labels(best_neighbors, accepted)


def build_microcluster_labels(
    item_count: int,
    accepted: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    directory: Path,
    check_cancel: Callable[[], None] | None = None,
) -> MicroclusterLabels:
    directory.mkdir(parents=True, exist_ok=True)
    best_neighbors = np.memmap(
        directory / "best_neighbors.i64",
        dtype=np.int64,
        mode="w+",
        shape=(item_count,),
    )
    best_scores = np.memmap(
        directory / "best_scores.f32",
        dtype=np.float32,
        mode="w+",
        shape=(item_count,),
    )
    best_neighbors[:] = -1
    best_scores[:] = -np.inf
    _find_best_neighbors(best_neighbors, best_scores, edge_blocks, check_cancel)
    labels = MicroclusterLabels.create(directory, item_count)
    labels.values[:] = _pair_labels(best_neighbors, accepted)
    labels.flush()
    return labels


def _find_best_neighbors(
    best_neighbors: np.ndarray,
    best_scores: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    check_cancel: Callable[[], None] | None,
) -> None:
    for block in edge_blocks:
        if check_cancel is not None:
            check_cancel()
        sources = np.concatenate((block.left_ids, block.right_ids))
        targets = np.concatenate((block.right_ids, block.left_ids))
        scores = np.concatenate((block.exact_scores, block.exact_scores))
        _update_best(best_neighbors, best_scores, sources, targets, scores)


def _update_best(
    best_neighbors: np.ndarray,
    best_scores: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
) -> None:
    order = np.lexsort((targets, -scores, sources))
    ordered_sources = sources[order]
    first = np.concatenate(
        (np.asarray([True]), ordered_sources[1:] != ordered_sources[:-1])
    )
    selected = order[first]
    source_ids = sources[selected]
    candidate_scores = scores[selected]
    candidate_neighbors = targets[selected]
    current_scores = best_scores[source_ids]
    current_neighbors = best_neighbors[source_ids]
    replace = (candidate_scores > current_scores) | (
        (candidate_scores == current_scores)
        & ((current_neighbors < 0) | (candidate_neighbors < current_neighbors))
    )
    best_scores[source_ids[replace]] = candidate_scores[replace]
    best_neighbors[source_ids[replace]] = candidate_neighbors[replace]


def _pair_labels(best_neighbors: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    item_count = len(best_neighbors)
    if len(accepted) != item_count:
        raise ValueError("accepted mask and microcluster labels must have equal length")
    row_ids = np.arange(item_count, dtype=np.int64)
    labels = np.where(accepted, row_ids, -1)
    valid = accepted & (best_neighbors >= 0) & (best_neighbors < item_count)
    safe_neighbors = np.where(valid, best_neighbors, 0)
    mutual = valid & (best_neighbors[safe_neighbors] == row_ids)
    pair_starts = mutual & (row_ids < best_neighbors)
    labels[best_neighbors[pair_starts]] = row_ids[pair_starts]
    return labels
