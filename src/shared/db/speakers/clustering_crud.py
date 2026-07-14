from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.db.speakers.models import (
    SpeakerClusteringArtifact,
    SpeakerClusteringRun,
    SpeakerClusterSummary,
    SpeakerEmbeddingRun,
)
from shared.db.speakers.schemas import (
    ClusterSummaryCreate,
    ClusteringArtifactCreate,
    ClusteringArtifactRole,
    ClusteringRunCreate,
    ClusteringOutcomeCounts,
    ClusteringRunState,
    EmbeddingRunState,
)


def create_clustering_run(
    session: Session, payload: ClusteringRunCreate
) -> SpeakerClusteringRun:
    existing = session.scalar(
        select(SpeakerClusteringRun).where(
            SpeakerClusteringRun.run_key == payload.run_key
        )
    )
    if existing is not None:
        _validate_run_identity(existing, payload)
        return existing
    embedding_run = session.get(SpeakerEmbeddingRun, payload.embedding_run_id)
    if embedding_run is None:
        raise KeyError(f"speaker embedding run not found: {payload.embedding_run_id}")
    if embedding_run.state != EmbeddingRunState.SEALED.value:
        raise ValueError(
            f"speaker embedding run {embedding_run.id} is {embedding_run.state}"
        )
    if payload.expected_count != embedding_run.expected_count:
        raise ValueError(
            f"clustering expected {payload.expected_count}, embedding run has "
            f"{embedding_run.expected_count}"
        )
    run = SpeakerClusteringRun(
        **payload.model_dump(),
        assignment_count=0,
        state=ClusteringRunState.OPEN.value,
    )
    session.add(run)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(
            select(SpeakerClusteringRun).where(
                SpeakerClusteringRun.run_key == payload.run_key
            )
        )
        if existing is None:
            raise
        _validate_run_identity(existing, payload)
        return existing
    session.refresh(run)
    return run


def register_clustering_artifact(
    session: Session,
    run_id: UUID,
    payload: ClusteringArtifactCreate,
) -> SpeakerClusteringArtifact:
    run = _locked_run(session, run_id)
    existing = session.scalar(
        select(SpeakerClusteringArtifact).where(
            SpeakerClusteringArtifact.run_id == run_id,
            SpeakerClusteringArtifact.artifact_id == payload.artifact_id,
        )
    )
    if existing is not None:
        stored = (existing.role, existing.ordinal, existing.row_count)
        incoming = (payload.role, payload.ordinal, payload.row_count)
        if stored != incoming:
            raise ValueError(
                f"artifact {payload.artifact_id} has different clustering metadata"
            )
        session.commit()
        return existing
    if run.state != ClusteringRunState.OPEN.value:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    artifact = SpeakerClusteringArtifact(run_id=run_id, **payload.model_dump())
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def list_clustering_artifacts(
    session: Session,
    run_id: UUID,
    role: ClusteringArtifactRole | None = None,
) -> list[SpeakerClusteringArtifact]:
    statement = select(SpeakerClusteringArtifact).where(
        SpeakerClusteringArtifact.run_id == run_id
    )
    if role is not None:
        statement = statement.where(SpeakerClusteringArtifact.role == role)
    return list(
        session.scalars(
            statement.order_by(
                SpeakerClusteringArtifact.role, SpeakerClusteringArtifact.ordinal
            )
        )
    )


def clear_open_clustering_artifacts(session: Session, run_id: UUID) -> list[UUID]:
    run = _locked_run(session, run_id)
    if run.state not in {
        ClusteringRunState.OPEN.value,
        ClusteringRunState.FAILED.value,
    }:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    artifact_ids = list(
        session.scalars(
            select(SpeakerClusteringArtifact.artifact_id).where(
                SpeakerClusteringArtifact.run_id == run_id
            )
        )
    )
    session.query(SpeakerClusterSummary).filter(
        SpeakerClusterSummary.run_id == run_id
    ).delete()
    run.assignment_count = 0
    run.outcome_counts = None
    run.failure_details = None
    run.prototype_artifact_id = None
    run.index_artifact_id = None
    run.completed_at = None
    run.state = ClusteringRunState.OPEN.value
    session.commit()
    return artifact_ids


def get_clustering_run(session: Session, run_id: UUID) -> SpeakerClusteringRun:
    run = session.get(SpeakerClusteringRun, run_id)
    if run is None:
        raise KeyError(f"speaker clustering run not found: {run_id}")
    return run


def replace_cluster_summaries(
    session: Session,
    run_id: UUID,
    payloads: list[ClusterSummaryCreate],
) -> list[SpeakerClusterSummary]:
    run = _locked_run(session, run_id)
    if run.state != ClusteringRunState.OPEN.value:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    session.query(SpeakerClusterSummary).filter(
        SpeakerClusterSummary.run_id == run_id
    ).delete()
    rows = [
        SpeakerClusterSummary(run_id=run_id, **payload.model_dump(mode="json"))
        for payload in payloads
    ]
    session.add_all(rows)
    session.commit()
    return rows


def append_cluster_summaries(
    session: Session,
    run_id: UUID,
    payloads: list[ClusterSummaryCreate],
) -> None:
    run = _locked_run(session, run_id)
    if run.state != ClusteringRunState.OPEN.value:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    session.add_all(
        SpeakerClusterSummary(run_id=run_id, **payload.model_dump(mode="json"))
        for payload in payloads
    )
    session.commit()


def fail_clustering_run(
    session: Session, run_id: UUID, details: dict[str, object]
) -> SpeakerClusteringRun:
    run = _locked_run(session, run_id)
    if run.state == ClusteringRunState.COMPLETED.value:
        raise ValueError(f"speaker clustering run {run_id} is completed")
    run.state = ClusteringRunState.FAILED.value
    run.failure_details = details
    session.commit()
    session.refresh(run)
    return run


def complete_clustering_run(
    session: Session,
    run_id: UUID,
    assignment_count: int,
    prototype_artifact_id: UUID,
    index_artifact_id: UUID,
    outcome_counts: ClusteringOutcomeCounts,
) -> SpeakerClusteringRun:
    run = _locked_run(session, run_id)
    if run.state != ClusteringRunState.OPEN.value:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    persisted_count = session.scalar(
        select(func.coalesce(func.sum(SpeakerClusteringArtifact.row_count), 0)).where(
            SpeakerClusteringArtifact.run_id == run_id,
            SpeakerClusteringArtifact.role == ClusteringArtifactRole.ASSIGNMENT.value,
        )
    )
    counted_outcomes = sum(outcome_counts.model_dump().values())
    if (
        persisted_count != assignment_count
        or assignment_count != run.expected_count
        or counted_outcomes != assignment_count
    ):
        raise ValueError(
            f"cannot complete clustering run {run_id}: persisted assignment count "
            f"{persisted_count}, declared {assignment_count}, "
            f"outcome count {counted_outcomes}, expected {run.expected_count}"
        )
    run.assignment_count = assignment_count
    run.prototype_artifact_id = prototype_artifact_id
    run.index_artifact_id = index_artifact_id
    run.outcome_counts = outcome_counts.model_dump()
    run.failure_details = None
    run.state = ClusteringRunState.COMPLETED.value
    run.completed_at = datetime.now(UTC)
    session.commit()
    session.refresh(run)
    return run


def _locked_run(session: Session, run_id: UUID) -> SpeakerClusteringRun:
    run = session.scalar(
        select(SpeakerClusteringRun)
        .where(SpeakerClusteringRun.id == run_id)
        .with_for_update()
    )
    if run is None:
        raise KeyError(f"speaker clustering run not found: {run_id}")
    return run


def _validate_run_identity(
    run: SpeakerClusteringRun,
    payload: ClusteringRunCreate,
) -> None:
    stored = (
        run.embedding_run_id,
        run.expected_count,
        run.index_factory,
        run.threshold_version,
        run.settings,
    )
    incoming = (
        payload.embedding_run_id,
        payload.expected_count,
        payload.index_factory,
        payload.threshold_version,
        payload.settings,
    )
    if stored != incoming:
        raise ValueError(
            f"clustering run key {payload.run_key} has different clustering identity"
        )
