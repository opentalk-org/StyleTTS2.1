from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
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
    ClusteringRunComplete,
    ClusteringRunCreate,
    ClusteringRunState,
    EmbeddingRunState,
)


def create_clustering_run(
    session: Session, payload: ClusteringRunCreate
) -> SpeakerClusteringRun:
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
    session.commit()
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
    role: str | None = None,
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


def complete_clustering_run(
    session: Session,
    run_id: UUID,
    payload: ClusteringRunComplete,
) -> SpeakerClusteringRun:
    run = _locked_run(session, run_id)
    if run.state != ClusteringRunState.OPEN.value:
        raise ValueError(f"speaker clustering run {run_id} is {run.state}")
    if payload.assignment_count != run.expected_count:
        raise ValueError(
            f"cannot complete clustering run {run_id}: expected {run.expected_count}, "
            f"assigned {payload.assignment_count}"
        )
    run.assignment_count = payload.assignment_count
    run.prototype_artifact_id = payload.prototype_artifact_id
    run.index_artifact_id = payload.index_artifact_id
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
