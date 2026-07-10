from __future__ import annotations

import secrets
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.datasets.models import dataset_audio_files
from shared.db.mos.models import MosComparison
from shared.db.mos.schemas import MosPair, MosRatingCreate, MosRatingUpdate
from shared.db.mos.mutations import delete_rating as delete_rating_mutation
from shared.db.mos.mutations import update_rating as update_rating_mutation


def sample_pair(session: Session, dataset_ids: Sequence[uuid.UUID]) -> MosPair:
    requested_ids = list(dict.fromkeys(dataset_ids))
    if not requested_ids:
        raise ValueError("MOS pair requires at least one dataset")
    eligible_ids = _eligible_dataset_ids(session, requested_ids)
    if not eligible_ids:
        raise ValueError("selected datasets do not contain two eligible audio files")
    dataset_id = secrets.choice(eligible_ids)
    audio_ids = _sample_audio_ids(session, dataset_id)
    assert len(audio_ids) == 2, f"eligible MOS dataset did not yield two audio files: {dataset_id}"
    return MosPair(
        dataset_id=dataset_id,
        audio_a=audio_crud.get_audio_file(session, audio_ids[0]),
        audio_b=audio_crud.get_audio_file(session, audio_ids[1]),
    )


def create_rating(session: Session, payload: MosRatingCreate) -> MosComparison:
    audio_ids = (payload.audio_a_id, payload.audio_b_id)
    membership_count = session.execute(
        select(func.count())
        .select_from(dataset_audio_files)
        .where(
            dataset_audio_files.c.dataset_id == payload.dataset_id,
            dataset_audio_files.c.audio_file_id.in_(audio_ids),
        )
    ).scalar_one()
    if membership_count != 2:
        raise ValueError("both MOS audio files must belong to the selected dataset")

    audio_a = audio_crud.get_audio_file(session, payload.audio_a_id)
    audio_b = audio_crud.get_audio_file(session, payload.audio_b_id)
    previous_score_a = audio_a.score
    previous_score_b = audio_b.score
    updated_at = datetime.now(UTC)
    audio_a.score = payload.score_a
    audio_a.updated_at = updated_at
    audio_b.score = payload.score_b
    audio_b.updated_at = updated_at
    comparison = MosComparison(
        **payload.model_dump(),
        previous_score_a=previous_score_a,
        previous_score_b=previous_score_b,
    )
    session.add(comparison)
    session.commit()
    session.refresh(comparison)
    return comparison


def list_comparisons(session: Session, dataset_id: uuid.UUID) -> list[MosComparison]:
    statement = (
        select(MosComparison)
        .where(MosComparison.dataset_id == dataset_id)
        .order_by(MosComparison.created_at.asc(), MosComparison.id.asc())
    )
    return list(session.execute(statement).scalars().all())


def count_comparisons(session: Session, dataset_id: uuid.UUID) -> int:
    statement = (
        select(func.count())
        .select_from(MosComparison)
        .where(MosComparison.dataset_id == dataset_id)
    )
    return session.execute(statement).scalar_one()


def iter_comparisons(session: Session, dataset_id: uuid.UUID) -> Iterator[MosComparison]:
    statement = (
        select(MosComparison)
        .where(MosComparison.dataset_id == dataset_id)
        .order_by(MosComparison.created_at.asc(), MosComparison.id.asc())
        .execution_options(yield_per=1_000)
    )
    yield from session.execute(statement).scalars()


def list_comparisons_page(
    session: Session,
    dataset_ids: Sequence[uuid.UUID],
    limit: int,
    offset: int,
) -> tuple[list[MosComparison], int]:
    requested_ids = list(dict.fromkeys(dataset_ids))
    if not requested_ids:
        raise ValueError("MOS history requires at least one dataset")
    filters = MosComparison.dataset_id.in_(requested_ids)
    total = session.execute(
        select(func.count()).select_from(MosComparison).where(filters)
    ).scalar_one()
    statement = (
        select(MosComparison)
        .where(filters)
        .order_by(MosComparison.created_at.desc(), MosComparison.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.execute(statement).scalars().all()), total


def comparison_audio_files(
    session: Session,
    comparisons: Sequence[MosComparison],
) -> dict[uuid.UUID, AudioFile]:
    audio_ids = {
        audio_id
        for comparison in comparisons
        for audio_id in (comparison.audio_a_id, comparison.audio_b_id)
    }
    if not audio_ids:
        return {}
    statement = select(AudioFile).where(AudioFile.id.in_(audio_ids))
    return {item.id: item for item in session.execute(statement).unique().scalars().all()}


def update_latest_rating(
    session: Session,
    comparison_id: uuid.UUID,
    payload: MosRatingUpdate,
) -> MosComparison:
    return update_rating_mutation(session, comparison_id, payload)


def undo_latest_rating(session: Session, comparison_id: uuid.UUID) -> None:
    delete_rating_mutation(session, comparison_id)


def _eligible_dataset_ids(session: Session, dataset_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    statement = (
        select(dataset_audio_files.c.dataset_id)
        .join(AudioFile, AudioFile.id == dataset_audio_files.c.audio_file_id)
        .where(
            dataset_audio_files.c.dataset_id.in_(dataset_ids),
            AudioFile.virtual.is_(False),
        )
        .group_by(dataset_audio_files.c.dataset_id)
        .having(func.count() >= 2)
    )
    return list(session.execute(statement).scalars().all())


def _sample_audio_ids(session: Session, dataset_id: uuid.UUID) -> list[uuid.UUID]:
    threshold = uuid.uuid4()
    statement = _dataset_audio_ids(dataset_id).where(AudioFile.id >= threshold).limit(2)
    audio_ids = list(session.execute(statement).scalars().all())
    if len(audio_ids) == 2:
        return audio_ids
    wrap = _dataset_audio_ids(dataset_id).where(AudioFile.id < threshold)
    if audio_ids:
        wrap = wrap.where(AudioFile.id != audio_ids[0])
    wrap = wrap.limit(2 - len(audio_ids))
    return audio_ids + list(session.execute(wrap).scalars().all())


def _dataset_audio_ids(dataset_id: uuid.UUID):
    return (
        select(AudioFile.id)
        .join(dataset_audio_files, dataset_audio_files.c.audio_file_id == AudioFile.id)
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.virtual.is_(False),
        )
        .order_by(AudioFile.id.asc())
    )
