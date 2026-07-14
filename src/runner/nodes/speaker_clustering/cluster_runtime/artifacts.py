from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from runner.nodes.speaker_clustering.cluster_runtime.assignment import AssignmentReason
from runner.nodes.speaker_clustering.prototypes import PrototypeStore
from runner.nodes.speaker_clustering.shards import EmbeddingQuality
from shared.db.speakers.schemas import SpeakerAssignmentOutcome


ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("audio_id", pa.string(), nullable=False),
        pa.field("duration_seconds", pa.float32(), nullable=False),
        pa.field("quality", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("cluster_id", pa.int64()),
        pa.field("best_cluster_id", pa.int64()),
        pa.field("second_cluster_id", pa.int64()),
        pa.field("best_score", pa.float32(), nullable=False),
        pa.field("second_score", pa.float32(), nullable=False),
        pa.field("margin", pa.float32(), nullable=False),
        pa.field("candidate_cluster_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("candidate_scores", pa.list_(pa.float32()), nullable=False),
        pa.field("threshold_version", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("true_label", pa.string()),
    ]
)


@dataclass(frozen=True)
class AssignmentRow:
    row_id: int
    segment_id: str
    audio_id: str
    duration_seconds: float
    quality: EmbeddingQuality
    outcome: SpeakerAssignmentOutcome
    cluster_id: int | None
    best_cluster_id: int | None
    second_cluster_id: int | None
    best_score: float
    second_score: float
    margin: float
    candidate_cluster_ids: list[int]
    candidate_scores: list[float]
    threshold_version: str
    reason: AssignmentReason
    true_label: str | None


def write_assignment_shards(
    blocks: Iterable[Sequence[AssignmentRow]],
    output_dir: Path,
    shard_rows: int,
) -> list[Path]:
    if shard_rows <= 0:
        raise ValueError(f"assignment shard_rows must be positive, got {shard_rows}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pending: list[AssignmentRow] = []
    for block in blocks:
        offset = 0
        while offset < len(block):
            consumed = min(shard_rows - len(pending), len(block) - offset)
            pending.extend(block[offset : offset + consumed])
            offset += consumed
            if len(pending) == shard_rows:
                paths.append(_write_assignment_file(pending, output_dir, len(paths)))
                pending = []
    if pending:
        paths.append(_write_assignment_file(pending, output_dir, len(paths)))
    return paths


def write_prototype_shards(
    prototypes: PrototypeStore,
    output_dir: Path,
    shard_rows: int,
    scan_rows: int,
) -> list[Path]:
    if shard_rows <= 0 or scan_rows <= 0:
        raise ValueError("prototype shard_rows and scan_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, prototypes.item_count, scan_rows):
        stop = min(start + scan_rows, prototypes.item_count)
        roots = np.flatnonzero(prototypes.member_counts[start:stop]) + start
        for offset in range(0, len(roots), shard_rows):
            selected = roots[offset : offset + shard_rows]
            path = output_dir / f"prototypes-{len(paths):08d}.parquet"
            pq.write_table(
                _prototype_table(prototypes, selected), path, compression="zstd"
            )
            paths.append(path)
    return paths


def _write_assignment_file(
    rows: list[AssignmentRow], output_dir: Path, ordinal: int
) -> Path:
    path = output_dir / f"assignments-{ordinal:08d}.parquet"
    table = pa.Table.from_pylist(
        [_assignment_mapping(row) for row in rows], ASSIGNMENT_SCHEMA
    )
    pq.write_table(table, path, compression="zstd")
    return path


def _assignment_mapping(row: AssignmentRow) -> dict[str, object]:
    return {
        "row_id": row.row_id,
        "segment_id": row.segment_id,
        "audio_id": row.audio_id,
        "duration_seconds": row.duration_seconds,
        "quality": row.quality.value,
        "outcome": row.outcome.value,
        "cluster_id": row.cluster_id,
        "best_cluster_id": row.best_cluster_id,
        "second_cluster_id": row.second_cluster_id,
        "best_score": row.best_score,
        "second_score": row.second_score,
        "margin": row.margin,
        "candidate_cluster_ids": row.candidate_cluster_ids,
        "candidate_scores": row.candidate_scores,
        "threshold_version": row.threshold_version,
        "reason": row.reason.value,
        "true_label": row.true_label,
    }


def _prototype_table(prototypes: PrototypeStore, roots: np.ndarray) -> pa.Table:
    dimension = prototypes.dimension
    vectors = np.asarray(prototypes.vectors[roots], dtype=np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()),
        dimension,
    )
    return pa.Table.from_arrays(
        [
            pa.array(roots, type=pa.int64()),
            vector_array,
            pa.array(prototypes.member_counts[roots], type=pa.int64()),
            pa.array(prototypes.duration_seconds[roots], type=pa.float64()),
            pa.array(prototypes.dispersion[roots], type=pa.float32()),
            pa.array(prototypes.suspicious[roots], type=pa.bool_()),
            pa.array(prototypes.exemplar_ids[roots], type=pa.int64()),
        ],
        names=[
            "cluster_id",
            "prototype",
            "member_count",
            "duration_seconds",
            "dispersion",
            "suspicious",
            "exemplar_id",
        ],
    )
