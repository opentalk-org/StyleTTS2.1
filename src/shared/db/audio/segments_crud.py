import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import SegmentCreate, SegmentUpdate
from shared.db.common import one


SegmentPayload = SegmentCreate | dict[str, Any]


def list_audio_segments(session: Session, audio_file_id: uuid.UUID) -> list[dict[str, Any]]:
    item = one(session, AudioFile, audio_file_id)
    return list(item.segments)


def list_audio_segments_bulk(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, list[dict[str, Any]]]:
    """Segments for many audio files in one query (avoids N+1 per-id lookups).

    Missing ids are simply absent from the result. Callers that need every id
    present should fall back to an empty list.
    """
    if not audio_file_ids:
        return {}
    unique_ids = list(dict.fromkeys(audio_file_ids))
    rows = session.execute(
        select(AudioFile.id, AudioFile.segments).where(AudioFile.id.in_(unique_ids))
    ).all()
    return {row.id: list(row.segments) for row in rows}


def create_segment(session: Session, audio_file_id: uuid.UUID, payload: SegmentCreate) -> dict[str, Any]:
    item = one(session, AudioFile, audio_file_id)
    segment = _segment_from_payload(payload)
    item.segments = [*item.segments, segment]
    item.updated_at = _now()
    session.commit()
    return segment


def replace_audio_segments(
    session: Session,
    audio_file_id: uuid.UUID,
    payloads: Sequence[SegmentPayload],
) -> list[dict[str, Any]]:
    item = one(session, AudioFile, audio_file_id)
    segments = [_segment_from_payload(payload) for payload in payloads]
    item.segments = segments
    item.updated_at = _now()
    session.commit()
    return segments


def bulk_replace_audio_segments(
    session: Session,
    payloads: dict[uuid.UUID, Sequence[SegmentPayload]],
) -> dict[uuid.UUID, list[dict[str, Any]]]:
    if not payloads:
        return {}
    out = {}
    now = _now()
    rows = []
    for audio_file_id, segment_payloads in payloads.items():
        segments = [_segment_from_payload(payload) for payload in segment_payloads]
        rows.append({"id": audio_file_id, "segments": segments, "updated_at": now})
        out[audio_file_id] = segments
    session.execute(update(AudioFile), rows)
    session.commit()
    return out


def update_segment_text(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    text: str,
) -> dict[str, Any]:
    return _update_segment_field(session, audio_file_id, segment_id, "text", text)


def update_segment_phonemes(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    phon: str,
) -> dict[str, Any]:
    return _update_segment_field(session, audio_file_id, segment_id, "phon", phon)


def update_segment(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    payload: SegmentUpdate,
) -> dict[str, Any]:
    item = one(session, AudioFile, audio_file_id)
    replacement = {"id": segment_id, **payload.model_dump(mode="json")}
    matched = False
    segments = []
    for segment in item.segments:
        if segment["id"] == segment_id:
            segments.append(replacement)
            matched = True
        else:
            segments.append(segment)
    if not matched:
        raise KeyError(f"Segment not found: {segment_id}")
    item.segments = segments
    item.updated_at = _now()
    session.commit()
    return replacement


def delete_segment(session: Session, audio_file_id: uuid.UUID, segment_id: str) -> None:
    item = one(session, AudioFile, audio_file_id)
    segments = [segment for segment in item.segments if segment["id"] != segment_id]
    if len(segments) == len(item.segments):
        raise KeyError(f"Segment not found: {segment_id}")
    item.segments = segments
    item.updated_at = _now()
    session.commit()


def _update_segment_field(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    field: str,
    value: str,
) -> dict[str, Any]:
    item = one(session, AudioFile, audio_file_id)
    updated_segment: dict[str, Any] | None = None
    segments = []
    for segment in item.segments:
        if segment["id"] == segment_id:
            updated_segment = {**segment, field: value}
            segments.append(updated_segment)
        else:
            segments.append(segment)
    if updated_segment is None:
        raise KeyError(f"Segment not found: {segment_id}")
    item.segments = segments
    item.updated_at = _now()
    session.commit()
    return updated_segment


def _segment_from_payload(payload: SegmentPayload) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else payload.model_dump(mode="json")
    segment = dict(raw)
    if "id" not in segment:
        segment["id"] = str(uuid.uuid4())
    return segment


def _now() -> datetime:
    return datetime.now(UTC)
