"""Disk-backed prototype aggregation keeps corpus-sized state outside the heap."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.cluster_runtime.prototype_blocks import (
    finalize_prototype_statistics,
    normalize_prototype_vectors,
)
from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock


class PrototypeStore:
    def __init__(
        self, directory: Path, item_count: int, dimension: int, mode: str
    ) -> None:
        self.item_count = item_count
        self.dimension = dimension
        self.vectors = np.memmap(
            directory / "prototypes.f32",
            dtype=np.float32,
            mode=mode,
            shape=(item_count, dimension),
        )
        self.member_counts = np.memmap(
            directory / "member_counts.i64",
            dtype=np.int64,
            mode=mode,
            shape=(item_count,),
        )
        self.duration_seconds = np.memmap(
            directory / "durations.f64",
            dtype=np.float64,
            mode=mode,
            shape=(item_count,),
        )
        self.dispersion = np.memmap(
            directory / "dispersion.f32",
            dtype=np.float32,
            mode=mode,
            shape=(item_count,),
        )
        self.suspicious = np.memmap(
            directory / "suspicious.bool",
            dtype=np.bool_,
            mode=mode,
            shape=(item_count,),
        )
        self.exemplar_ids = np.memmap(
            directory / "exemplar_ids.i64",
            dtype=np.int64,
            mode=mode,
            shape=(item_count,),
        )
        self.exemplar_scores = np.memmap(
            directory / "exemplar_scores.f32",
            dtype=np.float32,
            mode=mode,
            shape=(item_count,),
        )

    @classmethod
    def create(
        cls,
        directory: Path,
        item_count: int,
        dimension: int,
        block_rows: int,
        check_cancel: Callable[[], None] | None,
    ) -> PrototypeStore:
        directory.mkdir(parents=True, exist_ok=True)
        store = cls(directory, item_count, dimension, mode="w+")
        for start in range(0, item_count, block_rows):
            _check_cancel(check_cancel)
            stop = min(start + block_rows, item_count)
            store.vectors[start:stop] = 0.0
            store.member_counts[start:stop] = 0
            store.duration_seconds[start:stop] = 0.0
            store.dispersion[start:stop] = 0.0
            store.suspicious[start:stop] = False
            store.exemplar_ids[start:stop] = -1
            store.exemplar_scores[start:stop] = -np.inf
        return store

    def flush(self, check_cancel: Callable[[], None] | None = None) -> None:
        for values in (
            self.vectors,
            self.member_counts,
            self.duration_seconds,
            self.dispersion,
            self.suspicious,
            self.exemplar_ids,
            self.exemplar_scores,
        ):
            _check_cancel(check_cancel)
            values.flush()


def build_prototype_store(
    block_factory: Callable[[], Iterable[EmbeddingBlock]],
    labels: np.ndarray,
    directory: Path,
    item_count: int,
    dimension: int,
    max_members: int,
    max_dispersion: float,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> PrototypeStore:
    _check_cancel(check_cancel)
    store = PrototypeStore.create(
        directory, item_count, dimension, block_rows, check_cancel
    )
    for block in block_factory():
        if check_cancel is not None:
            check_cancel()
        valid = labels[block.row_ids] >= 0
        roots = labels[block.row_ids[valid]]
        np.add.at(store.vectors, roots, block.embeddings[valid])
        np.add.at(store.member_counts, roots, 1)
        np.add.at(store.duration_seconds, roots, block.duration_seconds[valid])
    normalize_prototype_vectors(
        store.vectors, store.member_counts, block_rows, check_cancel
    )
    for block in block_factory():
        if check_cancel is not None:
            check_cancel()
        valid = labels[block.row_ids] >= 0
        roots = labels[block.row_ids[valid]]
        vectors = _normalize_rows(block.embeddings[valid])
        scores = np.einsum("bd,bd->b", vectors, store.vectors[roots])
        distances = cosine_distances(scores)
        np.add.at(store.dispersion, roots, distances)
        _update_exemplars(
            store.exemplar_ids,
            store.exemplar_scores,
            roots,
            block.row_ids[valid],
            scores,
        )
    finalize_prototype_statistics(
        store.dispersion,
        store.member_counts,
        store.suspicious,
        max_members,
        max_dispersion,
        block_rows,
        check_cancel,
    )
    store.flush(check_cancel)
    return store


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0.0)


def cosine_distances(scores: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - scores, 0.0, 2.0)


def _update_exemplars(
    exemplar_ids: np.ndarray,
    exemplar_scores: np.ndarray,
    roots: np.ndarray,
    row_ids: np.ndarray,
    scores: np.ndarray,
) -> None:
    order = np.lexsort((row_ids, -scores, roots))
    ordered_roots = roots[order]
    first = np.concatenate(
        (np.asarray([True]), ordered_roots[1:] != ordered_roots[:-1])
    )
    selected = order[first]
    selected_roots = roots[selected]
    selected_scores = scores[selected]
    selected_ids = row_ids[selected]
    current_scores = exemplar_scores[selected_roots]
    current_ids = exemplar_ids[selected_roots]
    replace = (selected_scores > current_scores) | (
        (selected_scores == current_scores)
        & ((current_ids < 0) | (selected_ids < current_ids))
    )
    exemplar_scores[selected_roots[replace]] = selected_scores[replace]
    exemplar_ids[selected_roots[replace]] = selected_ids[replace]


def _check_cancel(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()
