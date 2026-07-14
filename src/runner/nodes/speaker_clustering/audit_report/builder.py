from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa

from runner.nodes.speaker_clustering.audit_metrics import (
    AssignmentAuditRow,
    ScoreSample,
    SpeakerAuditMetrics,
    compute_labeled_metrics,
    score_distribution,
)
from runner.nodes.speaker_clustering.audit_report.models import (
    AssignmentAuditDocument,
    AssignmentAuditInput,
)
from runner.nodes.speaker_clustering.audit_report.scans import AssignmentParquetScan
from runner.nodes.speaker_clustering.audit_report.selection import (
    select_listening_manifest,
)


LABELED_COLUMNS = ("segment_id", "cluster_id", "true_label")
MANIFEST_COLUMNS = (
    "segment_id",
    "audio_id",
    "duration_seconds",
    "cluster_id",
    "best_cluster_id",
    "second_cluster_id",
    "best_score",
    "second_score",
    "margin",
    "true_label",
)
REQUIRED_COLUMNS = frozenset((*MANIFEST_COLUMNS, "candidate_scores"))


def build_assignment_audit(
    assignment_paths: Sequence[Path],
    batch_rows: int,
    category_limit: int,
    check_cancel: Callable[[], None],
    report_progress: Callable[[int, int], None],
) -> AssignmentAuditDocument:
    if category_limit <= 0:
        raise ValueError("category_limit must be positive")
    scan = AssignmentParquetScan.create(
        paths=assignment_paths,
        batch_rows=batch_rows,
        required_columns=REQUIRED_COLUMNS,
        check_cancel=check_cancel,
        report_progress=report_progress,
    )
    labeled = compute_labeled_metrics(_labeled_rows(scan))
    metrics = SpeakerAuditMetrics(
        labeled=labeled,
        centroid_scores=score_distribution(_score_values(scan, "best_score")),
        second_scores=score_distribution(_score_values(scan, "second_score")),
        margins=score_distribution(_score_values(scan, "margin")),
        pair_scores=score_distribution(_candidate_scores(scan)),
    )
    manifest = select_listening_manifest(
        _manifest_rows(scan),
        frozenset(labeled.suspicious_cluster_ids),
        category_limit,
    )
    return AssignmentAuditDocument(
        total_rows=scan.total_rows,
        metrics=metrics,
        listening_manifest=manifest,
    )


def _labeled_rows(scan: AssignmentParquetScan) -> Iterator[AssignmentAuditRow]:
    for batch in scan.batches(LABELED_COLUMNS):
        segment_ids, cluster_ids, true_labels = _columns(batch, LABELED_COLUMNS)
        for segment_id, cluster_id, true_label in zip(
            segment_ids, cluster_ids, true_labels, strict=True
        ):
            yield AssignmentAuditRow(
                segment_id=str(segment_id),
                cluster_id=None if cluster_id is None else int(cluster_id),
                true_label=None if true_label is None else str(true_label),
                centroid_score=None,
                second_score=None,
            )


def _score_values(scan: AssignmentParquetScan, column: str) -> Iterator[ScoreSample]:
    for batch in scan.batches(("segment_id", column)):
        segment_ids = batch.column(0).to_pylist()
        for segment_id, value in zip(
            segment_ids, batch.column(1).to_pylist(), strict=True
        ):
            if value is not None:
                yield ScoreSample(f"{column}:{segment_id}", float(value))


def _candidate_scores(scan: AssignmentParquetScan) -> Iterator[ScoreSample]:
    for batch in scan.batches(("segment_id", "candidate_scores")):
        segment_ids = batch.column(0).to_pylist()
        for segment_id, values in zip(
            segment_ids, batch.column(1).to_pylist(), strict=True
        ):
            for index, value in enumerate(values):
                yield ScoreSample(f"candidate:{segment_id}:{index}", float(value))


def _manifest_rows(scan: AssignmentParquetScan) -> Iterator[AssignmentAuditInput]:
    for batch in scan.batches(MANIFEST_COLUMNS):
        for values in zip(*_columns(batch, MANIFEST_COLUMNS), strict=True):
            yield AssignmentAuditInput(
                segment_id=str(values[0]),
                audio_id=str(values[1]),
                duration_seconds=float(values[2]),
                cluster_id=_optional_int(values[3]),
                best_cluster_id=_optional_int(values[4]),
                second_cluster_id=_optional_int(values[5]),
                best_score=_optional_float(values[6]),
                second_score=_optional_float(values[7]),
                margin=_optional_float(values[8]),
                true_label=None if values[9] is None else str(values[9]),
            )


def _columns(batch: pa.RecordBatch, names: Sequence[str]) -> list[list[object]]:
    return [batch.column(name).to_pylist() for name in names]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
