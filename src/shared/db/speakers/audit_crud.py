from datetime import UTC, datetime
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
    report_artifact_id: UUID,
    listening_artifact_id: UUID,
    metrics: SpeakerAuditMetricsRecord,
) -> SpeakerClusterAudit:
    audit = _locked_audit(session, audit_id)
    metrics_payload = metrics.model_dump(mode="json")
    if audit.state == SpeakerAuditState.COMPLETED.value:
        stored = (
            audit.report_artifact_id,
            audit.listening_artifact_id,
            audit.metrics,
        )
        incoming = (report_artifact_id, listening_artifact_id, metrics_payload)
        if stored != incoming:
            raise ValueError(f"audit {audit_id} has different completion payload")
        session.commit()
        return audit
    if audit.state != SpeakerAuditState.OPEN.value:
        raise ValueError(f"speaker cluster audit {audit_id} is {audit.state}")
    audit.report_artifact_id = report_artifact_id
    audit.listening_artifact_id = listening_artifact_id
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
