import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from shared.db.common import many, one
from shared.db.voices.models import Voice
from shared.db.voices.schemas import VoiceCreate


def list_voices(session: Session) -> Sequence[Voice]:
    return many(session, Voice)


def create_voice(session: Session, payload: VoiceCreate) -> Voice:
    item = Voice(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def search_voices(session: Session, query: str, limit: int, offset: int) -> tuple[Sequence[Voice], int]:
    name_filter = Voice.name.ilike(f"%{query}%")
    rows = session.execute(
        select(Voice).where(name_filter).order_by(Voice.name).limit(limit).offset(offset)
    ).scalars().all()
    total = session.execute(select(func.count()).select_from(Voice).where(name_filter)).scalar_one()
    return rows, total


def rename_voice(session: Session, voice_id: uuid.UUID, name: str) -> Voice:
    item = one(session, Voice, voice_id)
    item.name = name
    session.commit()
    session.refresh(item)
    return item


def delete_voice(session: Session, voice_id: uuid.UUID) -> None:
    result = session.execute(delete(Voice).where(Voice.id == voice_id))
    if result.rowcount != 1:
        raise KeyError(f"Voice not found: {voice_id}")
    session.commit()
