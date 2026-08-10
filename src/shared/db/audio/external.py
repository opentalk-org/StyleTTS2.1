from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import ExternalAudioCreate
from shared.db.audio.segments import bulk_replace_audio_segments


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
            "score": payload.annotations.score,
            "language": payload.language,
            "style_prompt": payload.style_prompt,
            "voice_prompt": payload.voice_prompt,
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
    inserted = {payload.id: payload for payload in payloads if payload.id in inserted_ids}
    bulk_replace_audio_segments(
        session,
        {audio_id: payload.segments for audio_id, payload in inserted.items()},
        commit=False,
        fallback_accuracy={
            audio_id: payload.annotations.accuracy
            for audio_id, payload in inserted.items()
        },
    )
    session.commit()
    return len(inserted_ids)
