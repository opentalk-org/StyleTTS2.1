import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioStorageLocation


def audio_storage_locations(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, AudioStorageLocation]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return {}
    statement = (
        select(
            AudioFile.id,
            AudioFile.storage_kind,
            BucketFile.path,
            AudioFile.byte_offset,
            AudioFile.byte_length,
        )
        .outerjoin(BucketFile, AudioFile.bucket_file_id == BucketFile.id)
        .where(AudioFile.id.in_(ids))
    )
    rows = {row.id: row for row in session.execute(statement)}
    missing_ids = set(ids).difference(rows)
    if missing_ids:
        missing = sorted(str(audio_file_id) for audio_file_id in missing_ids)
        raise KeyError(f"Audio files not found: {missing}")
    locations = {}
    for audio_file_id in ids:
        row = rows[audio_file_id]
        if row.storage_kind != "packed":
            raise ValueError(
                f"Audio {audio_file_id} contains metadata only; "
                "no stored audio bytes are available"
            )
        if row.path is None:
            raise ValueError(f"packed audio has no object path: {audio_file_id}")
        locations[audio_file_id] = AudioStorageLocation(
            audio_file_id=audio_file_id,
            object_path=row.path,
            byte_offset=row.byte_offset,
            byte_length=row.byte_length,
        )
    return locations
