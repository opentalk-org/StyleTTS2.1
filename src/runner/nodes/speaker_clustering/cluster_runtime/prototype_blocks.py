from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
    labels: np.ndarray,
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
    for start in range(0, len(labels), block_rows):
        _check_cancel(check_cancel)
        block = labels[start : start + block_rows]
        valid = block >= 0
        rejected = np.zeros(len(block), dtype=np.bool_)
        rejected[valid] = suspicious[block[valid]]
        block[rejected] = -1
    if isinstance(labels, np.memmap):
        _check_cancel(check_cancel)
        labels.flush()


def _check_cancel(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()
