from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.db.speakers.models import SpeakerClusterAudit, SpeakerClusteringRun
from shared.db.speakers.schemas import (
    ClusteringRunState,
    SpeakerAuditCreate,
    SpeakerAuditMetricsRecord,
    SpeakerAuditState,
)


class SpeakerAuditApplyState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class SpeakerAuditApplyProgress:
    state: SpeakerAuditApplyState
    last_audio_id: UUID | None
    updated_audio_count: int
    total_audio_count: int
    error: str | None


def create_audit(
    session: Session, payload: SpeakerAuditCreate
) -> SpeakerClusterAudit:
    existing = _audit_for_identity(session, payload)
    if existing is not None:
        return existing
    cluster_run = session.get(SpeakerClusteringRun, payload.cluster_run_id)
    if cluster_run is None:
        raise KeyError(f"speaker clustering run not found: {payload.cluster_run_id}")
    if cluster_run.state != ClusteringRunState.COMPLETED.value:
        raise ValueError(f"speaker clustering run {cluster_run.id} is not completed")
    audit = SpeakerClusterAudit(
        **payload.model_dump(),
        state=SpeakerAuditState.OPEN.value,
    )
    session.add(audit)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = _audit_for_identity(session, payload)
        if existing is None:
            raise
        return existing
    session.refresh(audit)
    return audit


def complete_audit(
    session: Session,
    audit_id: UUID,
    review_id: UUID,
    metrics: SpeakerAuditMetricsRecord,
) -> SpeakerClusterAudit:
    audit = _locked_audit(session, audit_id)
    metrics_payload = metrics.model_dump(mode="json")
    if audit.state == SpeakerAuditState.COMPLETED.value:
        stored_metrics = SpeakerAuditMetricsRecord.model_validate(
            audit.metrics
        ).model_dump(mode="json")
        stored = (
            audit.review_id,
            stored_metrics,
        )
        incoming = (review_id, metrics_payload)
        if stored != incoming:
            raise ValueError(f"audit {audit_id} has different completion payload")
        session.commit()
        return audit
    if audit.state != SpeakerAuditState.OPEN.value:
        raise ValueError(f"speaker cluster audit {audit_id} is {audit.state}")
    audit.review_id = review_id
    audit.metrics = metrics_payload
    audit.state = SpeakerAuditState.COMPLETED.value
    audit.completed_at = datetime.now(UTC)
    session.commit()
    session.refresh(audit)
    return audit


def get_audit(session: Session, audit_id: UUID) -> SpeakerClusterAudit:
    audit = session.get(SpeakerClusterAudit, audit_id)
    if audit is None:
        raise KeyError(f"speaker cluster audit not found: {audit_id}")
    return audit


def get_audit_apply_progress(
    audit: SpeakerClusterAudit,
) -> SpeakerAuditApplyProgress | None:
    if audit.metrics is None or "apply" not in audit.metrics:
        return None
    value = audit.metrics["apply"]
    return SpeakerAuditApplyProgress(
        state=SpeakerAuditApplyState(value["state"]),
        last_audio_id=(
            None if value["last_audio_id"] is None else UUID(value["last_audio_id"])
        ),
        updated_audio_count=int(value["updated_audio_count"]),
        total_audio_count=int(value["total_audio_count"]),
        error=value["error"],
    )


def record_audit_apply_progress(
    session: Session,
    audit_id: UUID,
    progress: SpeakerAuditApplyProgress,
) -> None:
    audit = _locked_audit(session, audit_id)
    if audit.state != SpeakerAuditState.COMPLETED.value or audit.metrics is None:
        raise ValueError(f"speaker cluster audit {audit_id} is not completed")
    audit.metrics = {
        **audit.metrics,
        "apply": {
            "state": progress.state.value,
            "last_audio_id": (
                None
                if progress.last_audio_id is None
                else str(progress.last_audio_id)
            ),
            "updated_audio_count": progress.updated_audio_count,
            "total_audio_count": progress.total_audio_count,
            "error": progress.error,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    }
    session.commit()


def _audit_for_identity(
    session: Session, payload: SpeakerAuditCreate
) -> SpeakerClusterAudit | None:
    return session.scalar(
        select(SpeakerClusterAudit).where(
            SpeakerClusterAudit.cluster_run_id == payload.cluster_run_id,
            SpeakerClusterAudit.seed == payload.seed,
        )
    )


def _locked_audit(session: Session, audit_id: UUID) -> SpeakerClusterAudit:
    audit = session.scalar(
        select(SpeakerClusterAudit)
        .where(SpeakerClusterAudit.id == audit_id)
        .with_for_update()
    )
    if audit is None:
        raise KeyError(f"speaker cluster audit not found: {audit_id}")
    return audit
