from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import ExternalAudioCreate


def bulk_create_external_audio_files(
    session: Session,
    payloads: Sequence[ExternalAudioCreate],
) -> int:
    if not payloads:
        return 0
    values = [
        {
            "id": payload.id,
            "name": payload.name,
            "bucket_file_id": None,
            "byte_offset": 0,
            "byte_length": 0,
            "duration": payload.duration,
            "speaker_id": payload.annotations.speaker_id,
            "score": payload.annotations.score,
            "accuracy": payload.annotations.accuracy,
            "language": payload.language,
            "style_prompt": payload.style_prompt,
            "voice_prompt": payload.voice_prompt,
            "segments": payload.segments,
            "metadata": payload.annotations.metadata,
            "virtual": True,
            "storage_kind": "external",
            "storage_ref": payload.storage_ref.model_dump(mode="json"),
        }
        for payload in payloads
    ]
    statement = (
        insert(AudioFile.__table__)
        .values(values)
        .on_conflict_do_nothing(index_elements=["id"])
        .returning(AudioFile.id)
    )
    inserted_ids = set(session.execute(statement).scalars())
    session.commit()
    return len(inserted_ids)
