from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from shared.db.audio import crud as audio_crud
from shared.db.mos.models import MosComparison
from shared.db.mos.schemas import MosRatingUpdate


def update_rating(
    session: Session,
    comparison_id: uuid.UUID,
    payload: MosRatingUpdate,
) -> MosComparison:
    comparison = _get_comparison(session, comparison_id)
    if payload.preferred_audio_id not in (comparison.audio_a_id, comparison.audio_b_id):
        raise ValueError("MOS preferred audio must be a member of the pair")

    comparison.preferred_audio_id = payload.preferred_audio_id
    comparison.score_a = payload.score_a
    comparison.score_b = payload.score_b
    updated_at = datetime.now(UTC)

    _apply_audio_update(
        session=session,
        dataset_id=comparison.dataset_id,
        audio_id=comparison.audio_a_id,
        source_created_at=comparison.created_at,
        source_id=comparison.id,
        score=comparison.score_a,
        updated_at=updated_at,
    )
    _apply_audio_update(
        session=session,
        dataset_id=comparison.dataset_id,
        audio_id=comparison.audio_b_id,
        source_created_at=comparison.created_at,
        source_id=comparison.id,
        score=comparison.score_b,
        updated_at=updated_at,
    )

    session.commit()
    session.refresh(comparison)
    return comparison


def delete_rating(session: Session, comparison_id: uuid.UUID) -> None:
    comparison = _get_comparison(session, comparison_id)
    updated_at = datetime.now(UTC)

    _apply_audio_delete(
        session=session,
        dataset_id=comparison.dataset_id,
        audio_id=comparison.audio_a_id,
        source_created_at=comparison.created_at,
        source_id=comparison.id,
        previous_score=comparison.previous_score_a,
        updated_at=updated_at,
    )
    _apply_audio_delete(
        session=session,
        dataset_id=comparison.dataset_id,
        audio_id=comparison.audio_b_id,
        source_created_at=comparison.created_at,
        source_id=comparison.id,
        previous_score=comparison.previous_score_b,
        updated_at=updated_at,
    )

    session.delete(comparison)
    session.commit()


def _apply_audio_update(
    *,
    session: Session,
    dataset_id: uuid.UUID,
    audio_id: uuid.UUID,
    source_created_at: datetime,
    source_id: uuid.UUID,
    score: float,
    updated_at: datetime,
) -> None:
    next_comparison = _next_comparison_for_audio(
        session=session,
        dataset_id=dataset_id,
        audio_id=audio_id,
        source_created_at=source_created_at,
        source_id=source_id,
    )
    if next_comparison is None:
        audio = audio_crud.get_audio_file(session, audio_id)
        audio.score = score
        audio.updated_at = updated_at
        return

    _set_previous_score_for_audio(next_comparison, audio_id, score)


def _apply_audio_delete(
    *,
    session: Session,
    dataset_id: uuid.UUID,
    audio_id: uuid.UUID,
    source_created_at: datetime,
    source_id: uuid.UUID,
    previous_score: float | None,
    updated_at: datetime,
) -> None:
    next_comparison = _next_comparison_for_audio(
        session=session,
        dataset_id=dataset_id,
        audio_id=audio_id,
        source_created_at=source_created_at,
        source_id=source_id,
    )
    if next_comparison is not None:
        _set_previous_score_for_audio(next_comparison, audio_id, previous_score)
        return

    previous_comparison = _previous_comparison_for_audio(
        session=session,
        dataset_id=dataset_id,
        audio_id=audio_id,
        source_created_at=source_created_at,
        source_id=source_id,
    )
    audio = audio_crud.get_audio_file(session, audio_id)
    if previous_comparison is None:
        audio.score = previous_score
    else:
        audio.score = _score_for_audio(previous_comparison, audio_id)
    audio.updated_at = updated_at


def _set_previous_score_for_audio(
    comparison: MosComparison,
    audio_id: uuid.UUID,
    score: float | None,
) -> None:
    if comparison.audio_a_id == audio_id:
        comparison.previous_score_a = score
        return
    if comparison.audio_b_id == audio_id:
        comparison.previous_score_b = score
        return
    raise KeyError("audio is not part of this comparison")


def _get_comparison(session: Session, comparison_id: uuid.UUID) -> MosComparison:
    comparison = session.get(MosComparison, comparison_id)
    if comparison is None:
        raise KeyError("MOS comparison not found")
    return comparison


def _next_comparison_for_audio(
    *,
    session: Session,
    dataset_id: uuid.UUID,
    audio_id: uuid.UUID,
    source_created_at: datetime,
    source_id: uuid.UUID,
) -> MosComparison | None:
    statement = (
        select(MosComparison)
        .where(
            MosComparison.dataset_id == dataset_id,
            _audio_in_comparison(audio_id),
            or_(
                MosComparison.created_at > source_created_at,
                and_(MosComparison.created_at == source_created_at, MosComparison.id > source_id),
            ),
        )
        .order_by(MosComparison.created_at.asc(), MosComparison.id.asc())
        .limit(1)
    )
    return session.execute(statement).scalar_one_or_none()


def _previous_comparison_for_audio(
    *,
    session: Session,
    dataset_id: uuid.UUID,
    audio_id: uuid.UUID,
    source_created_at: datetime,
    source_id: uuid.UUID,
) -> MosComparison | None:
    statement = (
        select(MosComparison)
        .where(
            MosComparison.dataset_id == dataset_id,
            _audio_in_comparison(audio_id),
            or_(
                MosComparison.created_at < source_created_at,
                and_(MosComparison.created_at == source_created_at, MosComparison.id < source_id),
            ),
        )
        .order_by(MosComparison.created_at.desc(), MosComparison.id.desc())
        .limit(1)
    )
    return session.execute(statement).scalar_one_or_none()


def _score_for_audio(comparison: MosComparison, audio_id: uuid.UUID) -> float:
    if comparison.audio_a_id == audio_id:
        return comparison.score_a
    if comparison.audio_b_id == audio_id:
        return comparison.score_b
    raise KeyError("audio is not part of this comparison")


def _audio_in_comparison(audio_id: uuid.UUID):
    return (MosComparison.audio_a_id == audio_id) | (MosComparison.audio_b_id == audio_id)
