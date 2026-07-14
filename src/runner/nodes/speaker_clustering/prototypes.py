from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.cluster_runtime.prototype_blocks import (
    finalize_prototype_statistics,
    normalize_prototype_vectors,
)
from runner.nodes.speaker_clustering.cluster_runtime.support_pairs import (
    consolidate_labels as consolidate_labels,
    consolidate_labels_on_disk as consolidate_labels_on_disk,
)
from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock


@dataclass(frozen=True)
class PrototypeStatistics:
    vectors: np.ndarray
    member_counts: np.ndarray
    duration_seconds: np.ndarray
    dispersion: np.ndarray
    suspicious: np.ndarray
    exemplar_ids: np.ndarray


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


def prototype_statistics(
    vectors: np.ndarray,
    labels: np.ndarray,
    durations: np.ndarray,
    max_members: int,
    max_dispersion: float,
) -> PrototypeStatistics:
    item_count, dimension = vectors.shape
    sums = np.zeros((item_count, dimension), dtype=np.float32)
    counts = np.zeros(item_count, dtype=np.int64)
    total_durations = np.zeros(item_count, dtype=np.float64)
    valid = labels >= 0
    np.add.at(sums, labels[valid], vectors[valid])
    np.add.at(counts, labels[valid], 1)
    np.add.at(total_durations, labels[valid], durations[valid])
    prototypes = _normalize_prototypes(sums, counts)
    normalized_vectors = _normalize_rows(vectors)
    scores = np.zeros(item_count, dtype=np.float32)
    scores[valid] = np.einsum(
        "bd,bd->b", normalized_vectors[valid], prototypes[labels[valid]]
    )
    distances = 1.0 - scores
    dispersion_sums = np.zeros(item_count, dtype=np.float32)
    np.add.at(dispersion_sums, labels[valid], distances[valid])
    dispersion = np.divide(
        dispersion_sums,
        counts,
        out=np.zeros(item_count, dtype=np.float32),
        where=counts > 0,
    )
    suspicious = (counts > max_members) | ((counts > 0) & (dispersion > max_dispersion))
    exemplar_ids = np.full(item_count, -1, dtype=np.int64)
    exemplar_scores = np.full(item_count, -np.inf, dtype=np.float32)
    _update_exemplars(
        exemplar_ids,
        exemplar_scores,
        labels[valid],
        np.flatnonzero(valid),
        scores[valid],
    )
    return PrototypeStatistics(
        vectors=prototypes,
        member_counts=counts,
        duration_seconds=total_durations,
        dispersion=dispersion,
        suspicious=suspicious,
        exemplar_ids=exemplar_ids,
    )


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
        distances = 1.0 - scores
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
        labels,
        max_members,
        max_dispersion,
        block_rows,
        check_cancel,
    )
    store.flush(check_cancel)
    return store


def _normalize_prototypes(vectors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.empty_like(vectors, dtype=np.float32)
    normalized.fill(0.0)
    return np.divide(
        vectors,
        norms,
        out=normalized,
        where=(counts[:, np.newaxis] > 0) & (norms > 0.0),
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0.0)


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
