from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.models import SpeakerEmbeddingSetRef
from runner.nodes.speaker_clustering.shards import EmbeddingQuality
from shared.db import database_session
from shared.db.assets import crud as asset_crud


@dataclass(frozen=True)
class EmbeddingBlock:
    row_ids: np.ndarray
    segment_ids: np.ndarray
    audio_ids: np.ndarray
    duration_seconds: np.ndarray
    qualities: np.ndarray
    rejection_reasons: np.ndarray
    true_labels: np.ndarray
    embeddings: np.ndarray

    @classmethod
    def accepted_only(
        cls, row_ids: np.ndarray, embeddings: np.ndarray
    ) -> EmbeddingBlock:
        count = len(row_ids)
        empty = np.full(count, None, dtype=object)
        return cls(
            row_ids=np.asarray(row_ids, dtype=np.int64),
            segment_ids=empty.copy(),
            audio_ids=empty.copy(),
            duration_seconds=np.zeros(count, dtype=np.float32),
            qualities=np.full(count, EmbeddingQuality.ACCEPTED.value, dtype=object),
            rejection_reasons=empty.copy(),
            true_labels=empty,
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )

    @property
    def accepted_mask(self) -> np.ndarray:
        return self.qualities == EmbeddingQuality.ACCEPTED.value


def iter_embedding_blocks(
    embedding_set: SpeakerEmbeddingSetRef,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> Iterator[EmbeddingBlock]:
    with database_session() as session:
        paths = [
            asset_crud.get_extra_file_path(session, artifact_id)
            for artifact_id in embedding_set.artifact_ids
        ]
    yield from iter_embedding_paths(
        paths,
        dimension=embedding_set.dimension,
        block_rows=block_rows,
        check_cancel=check_cancel,
    )


def iter_embedding_paths(
    paths: Sequence[Path],
    dimension: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
) -> Iterator[EmbeddingBlock]:
    if block_rows <= 0:
        raise ValueError(f"embedding block_rows must be positive, got {block_rows}")
    row_offset = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        _validate_embedding_type(path, parquet.schema_arrow, dimension)
        for batch in parquet.iter_batches(batch_size=block_rows):
            if check_cancel is not None:
                check_cancel()
            count = batch.num_rows
            yield _embedding_block(batch, row_offset, dimension)
            row_offset += count


def _validate_embedding_type(path: Path, schema: pa.Schema, dimension: int) -> None:
    embedding_type = schema.field("embedding").type
    expected = pa.list_(pa.float16(), dimension)
    if embedding_type != expected:
        raise ValueError(
            f"embedding shard {path} has vector type {embedding_type}, expected {expected}"
        )


def _embedding_block(
    batch: pa.RecordBatch, row_offset: int, dimension: int
) -> EmbeddingBlock:
    embeddings = batch.column("embedding").values.to_numpy(zero_copy_only=False)
    return EmbeddingBlock(
        row_ids=np.arange(row_offset, row_offset + batch.num_rows, dtype=np.int64),
        segment_ids=_objects(batch, "segment_id"),
        audio_ids=_objects(batch, "audio_id"),
        duration_seconds=np.asarray(batch.column("duration_seconds"), dtype=np.float32),
        qualities=_objects(batch, "quality"),
        rejection_reasons=_objects(batch, "rejection_reason"),
        true_labels=_objects(batch, "true_label"),
        embeddings=np.asarray(embeddings, dtype=np.float16)
        .reshape(-1, dimension)
        .astype(np.float32),
    )


def _objects(batch: pa.RecordBatch, name: str) -> np.ndarray:
    return np.asarray(batch.column(name).to_pylist(), dtype=object)
