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

    def flush(self, check_cancel: Callable[[], None] | None = None) -> None:
        _check_cancel(check_cancel)
        self.values.flush()


def build_microcluster_labels(
    item_count: int,
    accepted: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    directory: Path,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> MicroclusterLabels:
    blocks = edge_blocks
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
    _fill_blocks(best_neighbors, -1, block_rows, check_cancel)
    _fill_blocks(best_scores, -np.inf, block_rows, check_cancel)
    _find_best_neighbors(best_neighbors, best_scores, blocks, check_cancel)
    labels = MicroclusterLabels.create(directory, item_count)
    _write_pair_labels(
        labels.values,
        best_neighbors,
        accepted,
        block_rows,
        check_cancel,
    )
    _check_cancel(check_cancel)
    best_neighbors.flush()
    _check_cancel(check_cancel)
    best_scores.flush()
    labels.flush(check_cancel)
    return labels


def _find_best_neighbors(
    best_neighbors: np.ndarray,
    best_scores: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    check_cancel: Callable[[], None] | None,
) -> int:
    block_rows = 0
    for block in edge_blocks:
        _check_cancel(check_cancel)
        sources = np.concatenate((block.left_ids, block.right_ids))
        targets = np.concatenate((block.right_ids, block.left_ids))
        scores = np.concatenate((block.exact_scores, block.exact_scores))
        block_rows = max(block_rows, len(sources))
        _update_best(best_neighbors, best_scores, sources, targets, scores)
    return block_rows


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


def _write_pair_labels(
    labels: np.ndarray,
    best_neighbors: np.ndarray,
    accepted: np.ndarray,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    item_count = len(best_neighbors)
    if len(accepted) != item_count:
        raise ValueError("accepted mask and microcluster labels must have equal length")
    for start in range(0, item_count, block_rows):
        _check_cancel(check_cancel)
        stop = min(start + block_rows, item_count)
        row_ids = np.arange(start, stop, dtype=np.int64)
        labels[start:stop] = np.where(accepted[start:stop], row_ids, -1)
    for start in range(0, item_count, block_rows):
        _check_cancel(check_cancel)
        stop = min(start + block_rows, item_count)
        row_ids = np.arange(start, stop, dtype=np.int64)
        neighbors = best_neighbors[start:stop]
        valid = accepted[start:stop] & (neighbors >= 0) & (neighbors < item_count)
        safe_neighbors = np.where(valid, neighbors, 0)
        mutual = valid & (best_neighbors[safe_neighbors] == row_ids)
        pair_starts = mutual & (row_ids < neighbors)
        labels[neighbors[pair_starts]] = row_ids[pair_starts]


def _fill_blocks(
    values: np.ndarray,
    value: int | float,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    for start in range(0, len(values), block_rows):
        _check_cancel(check_cancel)
        values[start : start + block_rows] = value


def _check_cancel(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()
