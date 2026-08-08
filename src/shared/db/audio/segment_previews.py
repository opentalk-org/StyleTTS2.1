import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioSegment


def list_audio_segment_previews_bulk(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
    limit: int,
) -> dict[uuid.UUID, list[dict[str, Any]]]:
    ids = list(dict.fromkeys(audio_file_ids))
    output = {audio_file_id: [] for audio_file_id in ids}
    rows = session.execute(
        select(
            AudioSegment.audio_file_id,
            AudioSegment.source_id,
            AudioSegment.start_seconds,
            AudioSegment.end_seconds,
            AudioSegment.text,
            AudioSegment.phon,
            AudioSegment.kind,
            AudioSegment.accuracy,
            AudioSegment.speaker_id,
            AudioSegment.metadata_,
        )
        .where(
            AudioSegment.audio_file_id.in_(ids),
            AudioSegment.position < limit,
        )
        .order_by(AudioSegment.audio_file_id, AudioSegment.position)
    ).all()
    for row in rows:
        annotations = {
            **row.metadata_["_source"]["annotations"],
            "accuracy": row.accuracy,
            "speaker_id": row.speaker_id,
        }
        output[row.audio_file_id].append(
            {
                "id": row.source_id,
                "start": row.start_seconds,
                "end": row.end_seconds,
                "text": row.text[:500],
                "phon": row.phon[:500],
                "type_": row.kind,
                "annotations": annotations,
                "alignment": None,
            }
        )
    return output


def list_audio_segment_previews(
    session: Session,
    audio_file_id: uuid.UUID,
    limit: int,
) -> list[dict[str, Any]]:
    return list_audio_segment_previews_bulk(session, [audio_file_id], limit)[audio_file_id]
