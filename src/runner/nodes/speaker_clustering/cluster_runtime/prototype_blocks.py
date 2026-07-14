from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PrototypeSelection:
    mask: np.memmap
    count: int

    def close(self) -> None:
        self.mask._mmap.close()


def build_prototype_selection(
    member_counts: np.ndarray,
    suspicious: np.ndarray,
    path: Path,
    min_members: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> PrototypeSelection:
    if len(member_counts) != len(suspicious):
        raise ValueError("prototype selection arrays must have equal length")
    if block_rows <= 0:
        raise ValueError("prototype selection block_rows must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.memmap(path, dtype=np.bool_, mode="w+", shape=(len(member_counts),))
    count = 0
    try:
        for start in range(0, len(member_counts), block_rows):
            _check_cancel(check_cancel)
            stop = min(start + block_rows, len(member_counts))
            selected = (member_counts[start:stop] >= min_members) & (
                ~suspicious[start:stop]
            )
            mask[start:stop] = selected
            count += int(np.count_nonzero(selected))
        _check_cancel(check_cancel)
        mask.flush()
        return PrototypeSelection(mask=mask, count=count)
    except BaseException:
        mask._mmap.close()
        raise


def create_prototype_neighbor_ids(
    path: Path,
    item_count: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> np.memmap:
    if block_rows <= 0:
        raise ValueError("prototype neighbor block_rows must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    neighbors = np.memmap(path, dtype=np.int64, mode="w+", shape=(item_count,))
    try:
        for start in range(0, item_count, block_rows):
            _check_cancel(check_cancel)
            neighbors[start : start + block_rows] = -1
        _check_cancel(check_cancel)
        neighbors.flush()
        return neighbors
    except BaseException:
        neighbors._mmap.close()
        raise


def normalize_prototype_vectors(
    vectors: np.ndarray,
    member_counts: np.ndarray,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    for start in range(0, len(vectors), block_rows):
        _check_cancel(check_cancel)
        stop = min(start + block_rows, len(vectors))
        block = vectors[start:stop]
        counts = member_counts[start:stop]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        valid = (counts[:, np.newaxis] > 0) & (norms > 0.0)
        np.divide(block, norms, out=block, where=valid)
        block[~valid[:, 0]] = 0.0


def finalize_prototype_statistics(
    dispersion: np.ndarray,
    member_counts: np.ndarray,
    suspicious: np.ndarray,
    max_members: int,
    max_dispersion: float,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    for start in range(0, len(member_counts), block_rows):
        _check_cancel(check_cancel)
        stop = min(start + block_rows, len(member_counts))
        counts = member_counts[start:stop]
        block_dispersion = dispersion[start:stop]
        np.divide(block_dispersion, counts, out=block_dispersion, where=counts > 0)
        suspicious[start:stop] = (counts > max_members) | (
            (counts > 0) & (block_dispersion > max_dispersion)
        )


def _check_cancel(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()
