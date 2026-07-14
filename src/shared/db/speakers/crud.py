from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.speakers.models import SpeakerEmbeddingRun, SpeakerEmbeddingShard
from shared.db.speakers.schemas import (
    EmbeddingRunCreate,
    EmbeddingRunState,
    EmbeddingShardCollection,
    EmbeddingShardCreate,
)


def create_embedding_run(
    session: Session, payload: EmbeddingRunCreate
) -> SpeakerEmbeddingRun:
    run = SpeakerEmbeddingRun(
        dataset_id=payload.dataset_id,
        expected_count=payload.expected_count,
        stored_count=0,
        dimension=payload.dimension,
        model_revision=payload.model_revision,
        preprocessing_version=payload.preprocessing_version,
        state=EmbeddingRunState.OPEN.value,
        failure_details=None,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def register_embedding_shard(
    session: Session, run_id: UUID, payload: EmbeddingShardCreate
) -> SpeakerEmbeddingShard:
    run = _locked_embedding_run(session, run_id)
    existing = _embedding_shard(session, run_id, payload.artifact_id)
    if run.state == EmbeddingRunState.SEALED.value and existing is not None:
        _validate_duplicate(existing, payload)
        session.commit()
        return existing
    if run.state != EmbeddingRunState.OPEN.value:
        raise ValueError(f"embedding run {run_id} is {run.state}")
    _validate_shard_identity(run, payload)
    if existing is not None:
        _validate_duplicate(existing, payload)
        session.commit()
        return existing
    shard = SpeakerEmbeddingShard(run_id=run_id, **payload.model_dump())
    run.stored_count += payload.row_count
    session.add(shard)
    session.commit()
    session.refresh(shard)
    return shard


def collect_embedding_shard(
    session: Session,
    run_id: UUID,
    payload: EmbeddingShardCreate,
) -> EmbeddingShardCollection:
    run = _locked_embedding_run(session, run_id)
    existing = _embedding_shard(session, run_id, payload.artifact_id)
    if run.state == EmbeddingRunState.SEALED.value:
        if existing is None:
            raise ValueError(
                f"embedding run {run_id} is sealed without artifact {payload.artifact_id}"
            )
        _validate_duplicate(existing, payload)
        return _collection_result(session, run, sealed_now=False)
    if run.state != EmbeddingRunState.OPEN.value:
        raise ValueError(f"embedding run {run_id} is {run.state}")

    _validate_shard_identity(run, payload)
    if existing is not None:
        _validate_duplicate(existing, payload)
    else:
        stored_count = run.stored_count + payload.row_count
        if stored_count > run.expected_count:
            raise ValueError(
                f"embedding run {run_id} would exceed expected count "
                f"{run.expected_count}: stored {run.stored_count}, incoming {payload.row_count}"
            )
        run.stored_count = stored_count
        session.add(SpeakerEmbeddingShard(run_id=run_id, **payload.model_dump()))
        session.flush()

    sealed_now = run.stored_count == run.expected_count
    if sealed_now:
        run.state = EmbeddingRunState.SEALED.value
        run.sealed_at = datetime.now(UTC)
    artifact_ids = [shard.artifact_id for shard in list_embedding_shards(session, run_id)]
    session.commit()
    session.refresh(run)
    return EmbeddingShardCollection(
        run_id=run.id,
        artifact_ids=artifact_ids,
        stored_count=run.stored_count,
        expected_count=run.expected_count,
        dimension=run.dimension,
        model_revision=run.model_revision,
        preprocessing_version=run.preprocessing_version,
        sealed_now=sealed_now,
    )


def get_embedding_run(session: Session, run_id: UUID) -> SpeakerEmbeddingRun:
    run = session.get(SpeakerEmbeddingRun, run_id)
    if run is None:
        raise KeyError(f"speaker embedding run not found: {run_id}")
    return run


def list_embedding_shards(
    session: Session, run_id: UUID
) -> list[SpeakerEmbeddingShard]:
    return list(
        session.scalars(
            select(SpeakerEmbeddingShard)
            .where(SpeakerEmbeddingShard.run_id == run_id)
            .order_by(SpeakerEmbeddingShard.created_at, SpeakerEmbeddingShard.id)
        )
    )


def seal_embedding_run(session: Session, run_id: UUID) -> SpeakerEmbeddingRun:
    run = _locked_embedding_run(session, run_id)
    if run.state != EmbeddingRunState.OPEN.value:
        raise ValueError(f"embedding run {run_id} is {run.state}")
    if run.stored_count != run.expected_count:
        raise ValueError(
            f"cannot seal embedding run {run_id}: expected {run.expected_count}, "
            f"stored {run.stored_count}"
        )
    run.state = EmbeddingRunState.SEALED.value
    run.sealed_at = datetime.now(UTC)
    session.commit()
    session.refresh(run)
    return run


def _locked_embedding_run(session: Session, run_id: UUID) -> SpeakerEmbeddingRun:
    run = session.scalar(
        select(SpeakerEmbeddingRun)
        .where(SpeakerEmbeddingRun.id == run_id)
        .with_for_update()
    )
    if run is None:
        raise KeyError(f"speaker embedding run not found: {run_id}")
    return run


def _embedding_shard(
    session: Session,
    run_id: UUID,
    artifact_id: UUID,
) -> SpeakerEmbeddingShard | None:
    return session.scalar(
        select(SpeakerEmbeddingShard).where(
            SpeakerEmbeddingShard.run_id == run_id,
            SpeakerEmbeddingShard.artifact_id == artifact_id,
        )
    )


def _collection_result(
    session: Session,
    run: SpeakerEmbeddingRun,
    sealed_now: bool,
) -> EmbeddingShardCollection:
    return EmbeddingShardCollection(
        run_id=run.id,
        artifact_ids=[
            shard.artifact_id for shard in list_embedding_shards(session, run.id)
        ],
        stored_count=run.stored_count,
        expected_count=run.expected_count,
        dimension=run.dimension,
        model_revision=run.model_revision,
        preprocessing_version=run.preprocessing_version,
        sealed_now=sealed_now,
    )


def _validate_shard_identity(
    run: SpeakerEmbeddingRun, payload: EmbeddingShardCreate
) -> None:
    identity = (payload.dimension, payload.model_revision, payload.preprocessing_version)
    expected = (run.dimension, run.model_revision, run.preprocessing_version)
    if identity != expected:
        raise ValueError("embedding shard identity does not match its run")


def _validate_duplicate(
    shard: SpeakerEmbeddingShard, payload: EmbeddingShardCreate
) -> None:
    stored = (
        shard.row_count,
        shard.dimension,
        shard.model_revision,
        shard.preprocessing_version,
    )
    incoming = (
        payload.row_count,
        payload.dimension,
        payload.model_revision,
        payload.preprocessing_version,
    )
    if stored != incoming:
        raise ValueError(f"artifact {payload.artifact_id} is registered with different metadata")
