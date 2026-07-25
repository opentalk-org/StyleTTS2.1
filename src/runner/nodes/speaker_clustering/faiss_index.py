from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock
from runner.nodes.speaker_clustering.reservoir import DeterministicVectorReservoir


@dataclass(frozen=True)
class FaissIndexSettings:
    index_factory: str = "IVF65536_HNSW32,Flat"
    training_rows: int = 1_000_000
    search_probes: int = 64
    random_seed: int = 0

    @classmethod
    def for_test(cls) -> FaissIndexSettings:
        return cls(
            index_factory="Flat",
            training_rows=1,
            search_probes=1,
            random_seed=0,
        )


@dataclass(frozen=True)
class CandidateSearch:
    scores: np.ndarray
    row_ids: np.ndarray


class SpeakerCandidateIndex:
    """Owns a normalized inner-product FAISS index with caller-defined row IDs."""

    def __init__(self, dimension: int, settings: FaissIndexSettings) -> None:
        if dimension <= 0:
            raise ValueError(f"FAISS dimension must be positive, got {dimension}")
        base = faiss.index_factory(
            dimension, settings.index_factory, faiss.METRIC_INNER_PRODUCT
        )
        self.dimension = dimension
        self.settings = settings
        self._index = faiss.IndexIDMap2(base)
        self._configure_search()

    @classmethod
    def load(cls, path: Path, settings: FaissIndexSettings) -> SpeakerCandidateIndex:
        index = faiss.read_index(str(path))
        instance = cls.__new__(cls)
        instance.dimension = index.d
        instance.settings = settings
        instance._index = index
        instance._configure_search()
        return instance

    @property
    def item_count(self) -> int:
        return self._index.ntotal

    @property
    def requires_training(self) -> bool:
        return not self._index.is_trained

    def train(self, vectors: np.ndarray) -> None:
        normalized = _normalized(vectors, self.dimension)
        if not self._index.is_trained:
            self._index.train(normalized)

    def add(self, row_ids: np.ndarray, vectors: np.ndarray) -> None:
        normalized = _normalized(vectors, self.dimension)
        ids = np.ascontiguousarray(row_ids, dtype=np.int64)
        if len(ids) != len(normalized):
            raise ValueError(
                f"FAISS received {len(ids)} IDs for {len(normalized)} vectors"
            )
        if not self._index.is_trained:
            raise ValueError("FAISS index must be trained before vectors are added")
        self._index.add_with_ids(normalized, ids)

    def search(self, vectors: np.ndarray, neighbors: int) -> CandidateSearch:
        if neighbors <= 0:
            raise ValueError(f"FAISS neighbors must be positive, got {neighbors}")
        normalized = _normalized(vectors, self.dimension)
        scores, row_ids = self._index.search(normalized, neighbors)
        return CandidateSearch(scores=scores, row_ids=row_ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def serialize(self) -> bytes:
        return faiss.serialize_index(self._index).tobytes()

    def _configure_search(self) -> None:
        parameters = faiss.ParameterSpace()
        parameter_name = (
            "efSearch" if self.settings.index_factory.startswith("HNSW") else "nprobe"
        )
        try:
            parameters.set_index_parameter(
                self._index, parameter_name, self.settings.search_probes
            )
        except RuntimeError:
            if self.settings.index_factory != "Flat":
                raise


def build_candidate_index_from_blocks(
    block_factory: Callable[[], Iterable[EmbeddingBlock]],
    dimension: int,
    settings: FaissIndexSettings,
    check_cancel: Callable[[], None] | None = None,
) -> SpeakerCandidateIndex:
    training_vectors = select_training_vectors(
        block_factory(),
        maximum_rows=settings.training_rows,
        random_seed=settings.random_seed,
        check_cancel=check_cancel,
    )
    index = SpeakerCandidateIndex(dimension, settings)
    if check_cancel is not None:
        check_cancel()
    index.train(training_vectors)
    if check_cancel is not None:
        check_cancel()
    for block in block_factory():
        if check_cancel is not None:
            check_cancel()
        accepted = block.accepted_mask
        if np.any(accepted):
            index.add(block.row_ids[accepted], block.embeddings[accepted])
    return index


def select_training_vectors(
    blocks: Iterable[EmbeddingBlock],
    maximum_rows: int,
    random_seed: int,
    check_cancel: Callable[[], None] | None = None,
) -> np.ndarray:
    reservoir = DeterministicVectorReservoir(maximum_rows, random_seed)
    for block in blocks:
        if check_cancel is not None:
            check_cancel()
        accepted = block.accepted_mask
        reservoir.add(block.row_ids[accepted], block.embeddings[accepted])
    return reservoir.result()


def _normalized(vectors: np.ndarray, dimension: int) -> np.ndarray:
    result = np.ascontiguousarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != dimension:
        raise ValueError(
            f"FAISS vectors have shape {result.shape}, expected (*, {dimension})"
        )
    if not np.isfinite(result).all():
        raise ValueError("FAISS vectors contain non-finite values")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("FAISS vectors contain zero-norm rows")
    return result / norms
