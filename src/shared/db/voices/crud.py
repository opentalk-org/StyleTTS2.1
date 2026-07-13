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
    return bulk_create_voices(session, [payload])[0]


def get_voices_by_names(
    session: Session,
    names: Sequence[str],
) -> dict[str, Voice]:
    requested = list(dict.fromkeys(names))
    if not requested:
        return {}
    statement = select(Voice).where(Voice.name.in_(requested))
    loaded = {voice.name: voice for voice in session.execute(statement).scalars().all()}
    return {name: loaded[name] for name in requested if name in loaded}


def bulk_create_voices(
    session: Session,
    payloads: Sequence[VoiceCreate],
) -> list[Voice]:
    names = [payload.name for payload in payloads]
    if len(set(names)) != len(names):
        raise ValueError("voice create payloads must have unique names")
    items = [Voice(**payload.model_dump()) for payload in payloads]
    if not items:
        return []
    session.add_all(items)
    session.commit()
    return items


def search_voices(session: Session, query: str, limit: int, offset: int) -> tuple[Sequence[Voice], int]:
    name_filter = Voice.name.ilike(f"%{query}%")
    rows = session.execute(
        select(Voice).where(name_filter).order_by(Voice.name).limit(limit).offset(offset)
    ).scalars().all()
    total = session.execute(select(func.count()).select_from(Voice).where(name_filter)).scalar_one()
    return rows, total


def search_voice_ids(session: Session, query: str) -> list[uuid.UUID]:
    name_filter = Voice.name.ilike(f"%{query}%")
    return list(session.execute(select(Voice.id).where(name_filter)).scalars().all())


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


def bulk_delete_voices(session: Session, voice_ids: Sequence[uuid.UUID]) -> None:
    if not voice_ids:
        return
    session.execute(delete(Voice).where(Voice.id.in_(voice_ids)))
    session.commit()
