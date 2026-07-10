from __future__ import annotations

import secrets
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.datasets.models import dataset_audio_files
from shared.db.mos.models import MosComparison
from shared.db.mos.schemas import MosPair, MosRatingCreate, MosRatingUpdate


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


def latest_comparison_id(session: Session) -> uuid.UUID | None:
    comparison = _latest_comparison(session)
    return comparison.id if comparison is not None else None


def update_latest_rating(
    session: Session,
    comparison_id: uuid.UUID,
    payload: MosRatingUpdate,
) -> MosComparison:
    comparison = _require_latest_comparison(session, comparison_id)
    if payload.preferred_audio_id not in (comparison.audio_a_id, comparison.audio_b_id):
        raise ValueError("MOS preferred audio must be a member of the pair")
    audio_a = audio_crud.get_audio_file(session, comparison.audio_a_id)
    audio_b = audio_crud.get_audio_file(session, comparison.audio_b_id)
    comparison.preferred_audio_id = payload.preferred_audio_id
    comparison.score_a = payload.score_a
    comparison.score_b = payload.score_b
    updated_at = datetime.now(UTC)
    audio_a.score = payload.score_a
    audio_a.updated_at = updated_at
    audio_b.score = payload.score_b
    audio_b.updated_at = updated_at
    session.commit()
    session.refresh(comparison)
    return comparison


def undo_latest_rating(session: Session, comparison_id: uuid.UUID) -> None:
    comparison = _require_latest_comparison(session, comparison_id)
    audio_a = audio_crud.get_audio_file(session, comparison.audio_a_id)
    audio_b = audio_crud.get_audio_file(session, comparison.audio_b_id)
    updated_at = datetime.now(UTC)
    audio_a.score = comparison.previous_score_a
    audio_a.updated_at = updated_at
    audio_b.score = comparison.previous_score_b
    audio_b.updated_at = updated_at
    session.delete(comparison)
    session.commit()


def _require_latest_comparison(session: Session, comparison_id: uuid.UUID) -> MosComparison:
    comparison = _latest_comparison(session)
    if comparison is None:
        raise KeyError("MOS comparison history is empty")
    if comparison.id != comparison_id:
        raise ValueError("only the newest MOS comparison can be changed or undone")
    return comparison


def _latest_comparison(session: Session) -> MosComparison | None:
    statement = (
        select(MosComparison)
        .order_by(MosComparison.created_at.desc(), MosComparison.id.desc())
        .limit(1)
    )
    return session.execute(statement).scalar_one_or_none()


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
