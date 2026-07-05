from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.common import many
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
