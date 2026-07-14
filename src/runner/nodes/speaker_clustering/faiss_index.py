from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import faiss
import numpy as np

from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock
from runner.nodes.speaker_clustering.shard_reader import iter_embedding_blocks
from runner.nodes.models import SpeakerEmbeddingSetRef


@dataclass(frozen=True)
class FaissIndexSettings:
    index_factory: str
    training_rows: int
    search_probes: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.training_rows <= 0 or self.search_probes <= 0:
            raise ValueError("FAISS training_rows and search_probes must be positive")


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
        try:
            parameters.set_index_parameter(
                self._index, "nprobe", self.settings.search_probes
            )
        except RuntimeError:
            if self.settings.index_factory != "Flat":
                raise


def build_candidate_index(
    embedding_set: SpeakerEmbeddingSetRef,
    settings: FaissIndexSettings,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> SpeakerCandidateIndex:
    block_factory = partial(
        iter_embedding_blocks,
        embedding_set,
        block_rows,
        check_cancel,
    )
    return build_candidate_index_from_blocks(
        block_factory,
        dimension=embedding_set.dimension,
        settings=settings,
        check_cancel=check_cancel,
    )


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
    index.train(training_vectors)
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
    if maximum_rows <= 0:
        raise ValueError(
            f"training sample maximum_rows must be positive, got {maximum_rows}"
        )
    selected_vectors: np.ndarray | None = None
    selected_ids = np.empty(0, dtype=np.int64)
    selected_priorities = np.empty(0, dtype=np.uint64)
    for block in blocks:
        if check_cancel is not None:
            check_cancel()
        accepted = block.accepted_mask
        vectors = block.embeddings[accepted]
        row_ids = block.row_ids[accepted]
        priorities = _stable_priorities(row_ids, random_seed)
        selected_vectors, selected_ids, selected_priorities = _retain_smallest(
            selected_vectors,
            selected_ids,
            selected_priorities,
            vectors,
            row_ids,
            priorities,
            maximum_rows,
        )
    if selected_vectors is None or not len(selected_vectors):
        raise ValueError("cannot train FAISS index without accepted speaker embeddings")
    order = np.lexsort((selected_ids, selected_priorities))
    return np.ascontiguousarray(selected_vectors[order], dtype=np.float32)


def _retain_smallest(
    selected_vectors: np.ndarray | None,
    selected_ids: np.ndarray,
    selected_priorities: np.ndarray,
    vectors: np.ndarray,
    row_ids: np.ndarray,
    priorities: np.ndarray,
    maximum_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_vectors = (
        vectors if selected_vectors is None else np.vstack((selected_vectors, vectors))
    )
    combined_ids = np.concatenate((selected_ids, row_ids))
    combined_priorities = np.concatenate((selected_priorities, priorities))
    if len(combined_ids) <= maximum_rows:
        return combined_vectors, combined_ids, combined_priorities
    retained = np.argpartition(combined_priorities, maximum_rows - 1)[:maximum_rows]
    return (
        combined_vectors[retained],
        combined_ids[retained],
        combined_priorities[retained],
    )


def _stable_priorities(row_ids: np.ndarray, random_seed: int) -> np.ndarray:
    values = np.asarray(row_ids, dtype=np.uint64) + np.uint64(random_seed)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


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
