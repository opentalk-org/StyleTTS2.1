from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from uuid import UUID

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.speaker_clustering.ecapa_runtime import ECAPA_EMBEDDING_DIMENSION


class EmbeddingQuality(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SpeakerEmbeddingRow:
    segment_id: str
    audio_id: UUID
    duration_seconds: float
    quality: EmbeddingQuality
    rejection_reason: str | None
    true_label: str | None
    embedding: np.ndarray | None


EMBEDDING_SHARD_SCHEMA = pa.schema(
    [
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("audio_id", pa.string(), nullable=False),
        pa.field("duration_seconds", pa.float32(), nullable=False),
        pa.field("quality", pa.string(), nullable=False),
        pa.field("rejection_reason", pa.string()),
        pa.field("true_label", pa.string()),
        pa.field("embedding", pa.list_(pa.float16(), ECAPA_EMBEDDING_DIMENSION)),
    ]
)


def write_embedding_shard(rows: list[SpeakerEmbeddingRow]) -> bytes:
    if not rows:
        raise ValueError("speaker embedding shard requires at least one row")
    values = np.zeros((len(rows), ECAPA_EMBEDDING_DIMENSION), dtype=np.float16)
    scalar_rows = []
    for index, row in enumerate(rows):
        scalar_rows.append(_arrow_row(row))
        if row.embedding is not None:
            values[index] = row.embedding.astype(np.float16)
    embedding_array = pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float16()),
        ECAPA_EMBEDDING_DIMENSION,
    )
    scalar_table = pa.Table.from_pylist(
        scalar_rows,
        schema=pa.schema(list(EMBEDDING_SHARD_SCHEMA)[:-1]),
    )
    table = scalar_table.append_column("embedding", embedding_array).cast(EMBEDDING_SHARD_SCHEMA)
    output = BytesIO()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue()


def read_embedding_shard(data: bytes) -> pa.Table:
    return pq.read_table(BytesIO(data), schema=EMBEDDING_SHARD_SCHEMA)


def _arrow_row(row: SpeakerEmbeddingRow) -> dict[str, object]:
    if row.embedding is not None:
        if row.embedding.shape != (ECAPA_EMBEDDING_DIMENSION,):
            raise ValueError(
                f"speaker embedding has shape {row.embedding.shape}, "
                f"expected ({ECAPA_EMBEDDING_DIMENSION},)"
            )
        if not np.isfinite(row.embedding).all():
            raise ValueError(f"speaker embedding is non-finite: {row.segment_id}")
    if row.quality is EmbeddingQuality.ACCEPTED and row.embedding is None:
        raise ValueError(f"accepted speaker embedding row has no vector: {row.segment_id}")
    if row.quality is EmbeddingQuality.REJECTED and row.embedding is not None:
        raise ValueError(f"rejected speaker embedding row has a vector: {row.segment_id}")
    return {
        "segment_id": row.segment_id,
        "audio_id": str(row.audio_id),
        "duration_seconds": row.duration_seconds,
        "quality": row.quality.value,
        "rejection_reason": row.rejection_reason,
        "true_label": row.true_label,
    }
