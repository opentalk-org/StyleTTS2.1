from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from shared.audio_annotations import AudioAnnotations
from shared.db.audio.clickhouse.models import AudioSegmentRecord


def segment_records(
    audio_id: UUID, payloads: Sequence[dict[str, Any]], updated_at: datetime
) -> list[AudioSegmentRecord]:
    records = []
    for position, payload in enumerate(payloads):
        annotations = AudioAnnotations.model_validate(payload["annotations"])
        records.append(
            AudioSegmentRecord(
                id=str(payload["id"]),
                audio_file_id=audio_id,
                updated_at=updated_at,
                position=position,
                start_seconds=float(payload["start"]),
                end_seconds=float(payload["end"]),
                text=str(payload["text"]),
                phon=str(payload["phon"]),
                kind=str(payload["type"]),
                accuracy=annotations.accuracy,
                speaker_id=annotations.speaker_id,
                metadata=annotations.metadata,
                alignment=payload["alignment"],
            )
        )
    return records
