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
from shared.db.mos.schemas import MosPair, MosRatingCreate


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
    updated_at = datetime.now(UTC)
    audio_a.score = payload.score_a
    audio_a.updated_at = updated_at
    audio_b.score = payload.score_b
    audio_b.updated_at = updated_at
    comparison = MosComparison(**payload.model_dump())
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
