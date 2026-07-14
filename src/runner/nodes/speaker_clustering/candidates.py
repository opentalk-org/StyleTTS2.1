from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runner.nodes.speaker_clustering.faiss_index import SpeakerCandidateIndex
from runner.nodes.speaker_clustering.shard_reader import EmbeddingBlock


@dataclass(frozen=True)
class RankedCandidates:
    row_ids: np.ndarray
    scores: np.ndarray


class CanonicalEmbeddingStore:
    def __init__(
        self, directory: Path, item_count: int, dimension: int, mode: str
    ) -> None:
        if item_count <= 0 or dimension <= 0:
            raise ValueError("canonical embedding store dimensions must be positive")
        self.item_count = item_count
        self.dimension = dimension
        self.vectors = np.memmap(
            directory / "vectors.f16",
            dtype=np.float16,
            mode=mode,
            shape=(item_count, dimension),
        )
        self.accepted = np.memmap(
            directory / "accepted.bool",
            dtype=np.bool_,
            mode=mode,
            shape=(item_count,),
        )

    @classmethod
    def create(
        cls, directory: Path, item_count: int, dimension: int
    ) -> CanonicalEmbeddingStore:
        directory.mkdir(parents=True, exist_ok=True)
        store = cls(directory, item_count, dimension, mode="w+")
        store.vectors[:] = 0.0
        store.accepted[:] = False
        return store

    def write(self, block: EmbeddingBlock) -> None:
        if np.any(block.row_ids < 0) or np.any(block.row_ids >= self.item_count):
            raise ValueError("canonical embedding row ID is outside the store")
        self.vectors[block.row_ids] = block.embeddings.astype(np.float16)
        self.accepted[block.row_ids] = block.accepted_mask

    def flush(self) -> None:
        self.vectors.flush()
        self.accepted.flush()


class CandidateMatrix:
    def __init__(
        self, directory: Path, item_count: int, neighbors: int, mode: str
    ) -> None:
        if item_count <= 0 or neighbors <= 0:
            raise ValueError("candidate matrix dimensions must be positive")
        self.item_count = item_count
        self.neighbors = neighbors
        self.row_ids = np.memmap(
            directory / "row_ids.i64",
            dtype=np.int64,
            mode=mode,
            shape=(item_count, neighbors),
        )
        self.scores = np.memmap(
            directory / "scores.f32",
            dtype=np.float32,
            mode=mode,
            shape=(item_count, neighbors),
        )

    @classmethod
    def create(
        cls, directory: Path, item_count: int, neighbors: int
    ) -> CandidateMatrix:
        directory.mkdir(parents=True, exist_ok=True)
        matrix = cls(directory, item_count, neighbors, mode="w+")
        matrix.row_ids[:] = -1
        matrix.scores[:] = -np.inf
        return matrix

    def write(
        self, query_ids: np.ndarray, row_ids: np.ndarray, scores: np.ndarray
    ) -> None:
        expected = (len(query_ids), self.neighbors)
        if row_ids.shape != expected or scores.shape != expected:
            raise ValueError(
                f"candidate result shapes are {row_ids.shape}/{scores.shape}, expected {expected}"
            )
        self.row_ids[query_ids] = row_ids
        self.scores[query_ids] = scores

    def flush(self) -> None:
        self.row_ids.flush()
        self.scores.flush()


@dataclass(frozen=True)
class ReciprocalCandidate:
    score: float
    rank: int


class ReciprocalCandidateLookup:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)

    @classmethod
    def create(
        cls,
        path: Path,
        candidates: CandidateMatrix,
        block_rows: int,
        check_cancel: Callable[[], None] | None = None,
    ) -> ReciprocalCandidateLookup:
        lookup = cls(path)
        lookup.connection.execute("PRAGMA journal_mode=OFF")
        lookup.connection.execute("PRAGMA synchronous=OFF")
        lookup.connection.execute(
            """
            CREATE TABLE candidates (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                score REAL NOT NULL,
                rank INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            ) WITHOUT ROWID
            """
        )
        statement = "INSERT INTO candidates VALUES (?, ?, ?, ?)"
        for start in range(0, candidates.item_count, block_rows):
            if check_cancel is not None:
                check_cancel()
            stop = min(start + block_rows, candidates.item_count)
            row_ids = np.asarray(candidates.row_ids[start:stop])
            scores = np.asarray(candidates.scores[start:stop])
            rows, ranks = np.nonzero(row_ids >= 0)
            records = (
                (
                    start + int(row),
                    int(row_ids[row, rank]),
                    float(scores[row, rank]),
                    int(rank) + 1,
                )
                for row, rank in zip(rows, ranks, strict=True)
            )
            lookup.connection.executemany(statement, records)
        lookup.connection.commit()
        return lookup

    def get(self, source_id: int, target_id: int) -> ReciprocalCandidate | None:
        row = self.connection.execute(
            "SELECT score, rank FROM candidates WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()
        if row is None:
            return None
        return ReciprocalCandidate(score=float(row[0]), rank=int(row[1]))

    def close(self) -> None:
        self.connection.close()


def build_canonical_store(
    blocks: Iterable[EmbeddingBlock],
    directory: Path,
    item_count: int,
    dimension: int,
    check_cancel: Callable[[], None] | None = None,
) -> CanonicalEmbeddingStore:
    store = CanonicalEmbeddingStore.create(directory, item_count, dimension)
    for block in blocks:
        if check_cancel is not None:
            check_cancel()
        store.write(block)
    store.flush()
    return store


def rerank_candidate_blocks(
    index: SpeakerCandidateIndex,
    blocks: Iterable[EmbeddingBlock],
    canonical: CanonicalEmbeddingStore,
    output: CandidateMatrix,
    threshold: float,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    for block in blocks:
        if check_cancel is not None:
            check_cancel()
        accepted = block.accepted_mask
        if not np.any(accepted):
            continue
        query_ids = block.row_ids[accepted]
        query_vectors = block.embeddings[accepted]
        proposed = index.search(query_vectors, output.neighbors + 1)
        reranked = exact_rerank(
            query_vectors,
            query_ids,
            proposed.row_ids,
            canonical.vectors,
            canonical.accepted,
            threshold,
            keep=output.neighbors,
        )
        output.write(query_ids, reranked.row_ids, reranked.scores)
    output.flush()


def exact_rerank(
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    canonical_vectors: np.ndarray,
    accepted: np.ndarray,
    threshold: float,
    keep: int | None = None,
) -> RankedCandidates:
    if candidate_ids.ndim != 2 or len(candidate_ids) != len(query_ids):
        raise ValueError("candidate IDs must be a matrix aligned with query IDs")
    retained = candidate_ids.shape[1] if keep is None else keep
    if retained <= 0 or retained > candidate_ids.shape[1]:
        raise ValueError(f"exact reranking keep={retained} is invalid")
    valid = (candidate_ids >= 0) & (candidate_ids < len(canonical_vectors))
    safe_ids = np.where(valid, candidate_ids, 0)
    valid &= np.asarray(accepted[safe_ids], dtype=np.bool_)
    valid &= safe_ids != query_ids[:, np.newaxis]
    queries = _normalized(query_vectors)
    candidates = _normalized_rows(
        np.asarray(canonical_vectors[safe_ids], dtype=np.float32)
    )
    scores = np.einsum("bd,bkd->bk", queries, candidates, optimize=True)
    scores[~valid | (scores < threshold)] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")[:, :retained]
    ranked_scores = np.take_along_axis(scores, order, axis=1).astype(np.float32)
    ranked_ids = np.take_along_axis(candidate_ids, order, axis=1).astype(np.int64)
    ranked_ids[~np.isfinite(ranked_scores)] = -1
    return RankedCandidates(row_ids=ranked_ids, scores=ranked_scores)


def _normalized(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norms == 0.0):
        raise ValueError("exact reranking received invalid query vectors")
    return result / norms


def _normalized_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms
