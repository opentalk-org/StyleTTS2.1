from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shared.db.audio.clickhouse.conversion import segment_records
from shared.db.audio.clickhouse.files import create_audio_files, get_audio_files
from shared.db.audio.clickhouse.models import AudioFileRecord, StorageKind
from shared.db.audio.clickhouse.segments import replace_audio_segments
from shared.db.audio.schemas import ExternalAudioCreate


def bulk_create_external_audio_files(
    _session: Session, payloads: Sequence[ExternalAudioCreate]
) -> int:
    if not payloads:
        return 0
    existing = {
        item.id for item in get_audio_files([payload.id for payload in payloads])
    }
    inserted = [payload for payload in payloads if payload.id not in existing]
    now = datetime.now(UTC)
    records = [
        AudioFileRecord(
            id=payload.id,
            updated_at=now,
            name=payload.name,
            bucket_file_id=None,
            byte_offset=0,
            duration=payload.duration,
            byte_length=0,
            score=payload.annotations.score,
            language=payload.language,
            style_prompt=payload.style_prompt,
            voice_prompt=payload.voice_prompt,
            virtual=True,
            storage_kind=StorageKind.EXTERNAL,
            storage_ref=payload.storage_ref.model_dump(mode="json"),
            metadata=payload.annotations.metadata,
        )
        for payload in inserted
    ]
    create_audio_files(records)
    for payload in inserted:
        replace_audio_segments(
            payload.id, segment_records(payload.id, payload.segments, now)
        )
    return len(inserted)
