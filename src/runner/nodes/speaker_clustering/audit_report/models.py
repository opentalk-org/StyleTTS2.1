from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from runner.nodes.speaker_clustering.audit_metrics import SpeakerAuditMetrics


@dataclass(frozen=True)
class AssignmentAuditInput:
    segment_id: str
    audio_id: str
    duration_seconds: float
    cluster_id: int | None
    best_cluster_id: int | None
    second_cluster_id: int | None
    best_score: float | None
    second_score: float | None
    margin: float | None
    true_label: str | None


class ListeningEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    audio_id: str
    duration_seconds: float = Field(ge=0.0)
    cluster_id: int | None
    best_cluster_id: int | None
    second_cluster_id: int | None
    true_label: str | None
    best_score: float | None
    second_score: float | None
    margin: float | None


class ListeningManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    worst_within_cluster: tuple[ListeningEntry, ...]
    closest_cross_cluster: tuple[ListeningEntry, ...]
    low_margin_boundaries: tuple[ListeningEntry, ...]
    suspicious_labeled_merges: tuple[ListeningEntry, ...]


class AssignmentAuditDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_rows: int = Field(ge=0)
    metrics: SpeakerAuditMetrics
    listening_manifest: ListeningManifest


class AssignmentAuditBuildResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    json_report_path: Path
    html_report_path: Path
    listening_manifest_path: Path
