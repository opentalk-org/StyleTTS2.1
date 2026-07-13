import math
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile


def bulk_update_audio_scores(
    session: Session,
    scores: dict[uuid.UUID, float],
) -> dict[uuid.UUID, AudioFile]:
    if not scores:
        raise ValueError("audio score update requires at least one item")
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("audio scores must be finite")
    statement = select(AudioFile).where(AudioFile.id.in_(scores))
    items = {item.id: item for item in session.execute(statement).unique().scalars().all()}
    missing_ids = set(scores).difference(items)
    if missing_ids:
        raise KeyError(f"Audio files not found: {sorted(str(audio_id) for audio_id in missing_ids)}")
    updated_at = datetime.now(UTC)
    for audio_id, score in scores.items():
        items[audio_id].score = score
        items[audio_id].updated_at = updated_at
    session.commit()
    return {audio_id: items[audio_id] for audio_id in scores}
