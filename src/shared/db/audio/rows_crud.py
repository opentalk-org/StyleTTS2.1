import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile


def get_audio_files_bulk(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, AudioFile]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return {}
    statement = select(AudioFile).where(AudioFile.id.in_(ids))
    loaded = {
        item.id: item
        for item in session.execute(statement).unique().scalars().all()
    }
    missing_ids = set(ids).difference(loaded)
    if missing_ids:
        missing = sorted(str(audio_file_id) for audio_file_id in missing_ids)
        raise KeyError(f"Audio files not found: {missing}")
    return {audio_file_id: loaded[audio_file_id] for audio_file_id in ids}
