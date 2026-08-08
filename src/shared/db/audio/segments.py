import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from shared.db.audio.models import Alignment, AudioFile, AudioSegment
from shared.db.audio.schemas import SegmentCreate, SegmentUpdate
from shared.db.common import one


SegmentPayload = SegmentCreate | dict[str, Any]


def list_audio_segments(session: Session, audio_file_id: uuid.UUID) -> list[dict[str, Any]]:
    one(session, AudioFile, audio_file_id)
    rows = session.execute(
        select(AudioSegment)
        .where(AudioSegment.audio_file_id == audio_file_id)
        .order_by(AudioSegment.position)
    ).unique().scalars()
    return [row.as_payload() for row in rows]


def list_audio_segments_bulk(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, list[dict[str, Any]]]:
    ids = list(dict.fromkeys(audio_file_ids))
    output = {audio_file_id: [] for audio_file_id in ids}
    rows = session.execute(
        select(AudioSegment)
        .where(AudioSegment.audio_file_id.in_(ids))
        .order_by(AudioSegment.audio_file_id, AudioSegment.position)
    ).unique().scalars()
    for row in rows:
        output[row.audio_file_id].append(row.as_payload())
    return output


def clear_audio_segment_phonemes_batch(session: Session, batch_size: int) -> int:
    target_ids = (
        select(AudioSegment.id)
        .where(AudioSegment.phon != "")
        .order_by(AudioSegment.id)
        .limit(batch_size)
        .scalar_subquery()
    )
    result = session.execute(
        update(AudioSegment).where(AudioSegment.id.in_(target_ids)).values(phon="")
    )
    session.commit()
    return int(result.rowcount)


def create_segment(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: SegmentCreate,
) -> dict[str, Any]:
    one(session, AudioFile, audio_file_id)
    position = session.execute(
        select(func.coalesce(func.max(AudioSegment.position), -1) + 1).where(
            AudioSegment.audio_file_id == audio_file_id
        )
    ).scalar_one()
    row = _segment_row(audio_file_id, int(position), payload, None)
    session.add(row)
    _touch_audio(session, audio_file_id)
    session.commit()
    return row.as_payload()


def replace_audio_segments(
    session: Session,
    audio_file_id: uuid.UUID,
    payloads: Sequence[SegmentPayload],
) -> list[dict[str, Any]]:
    return bulk_replace_audio_segments(session, {audio_file_id: payloads})[audio_file_id]


def bulk_replace_audio_segments(
    session: Session,
    payloads: dict[uuid.UUID, Sequence[SegmentPayload]],
    commit: bool = True,
    fallback_accuracy: dict[uuid.UUID, float | None] | None = None,
) -> dict[uuid.UUID, list[dict[str, Any]]]:
    if not payloads:
        return {}
    ids = list(payloads)
    existing = set(
        session.execute(select(AudioFile.id).where(AudioFile.id.in_(ids))).scalars()
    )
    missing = set(ids).difference(existing)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(str(item) for item in missing)}")
    segment_ids = select(AudioSegment.id).where(AudioSegment.audio_file_id.in_(ids))
    session.execute(delete(Alignment).where(Alignment.segment_id.in_(segment_ids)))
    session.execute(delete(AudioSegment).where(AudioSegment.audio_file_id.in_(ids)))
    rows_by_audio: dict[uuid.UUID, list[AudioSegment]] = {}
    for audio_file_id, segment_payloads in payloads.items():
        fallback = fallback_accuracy[audio_file_id] if fallback_accuracy is not None else None
        rows = [
            _segment_row(audio_file_id, position, payload, fallback)
            for position, payload in enumerate(segment_payloads)
        ]
        rows_by_audio[audio_file_id] = rows
        session.add_all(rows)
    now = datetime.now(UTC)
    session.execute(
        update(AudioFile)
        .where(AudioFile.id.in_(ids))
        .values(
            updated_at=now,
            segment_count=func.coalesce(
                select(func.count())
                .select_from(AudioSegment)
                .where(AudioSegment.audio_file_id == AudioFile.id)
                .correlate(AudioFile)
                .scalar_subquery(),
                0,
            ),
        )
    )
    session.flush()
    for audio_file_id in ids:
        item = session.identity_map.get(session.identity_key(AudioFile, audio_file_id))
        if item is not None:
            session.expire(item, ["segment_rows"])
    if commit:
        session.commit()
    return {
        audio_file_id: [row.as_payload() for row in rows]
        for audio_file_id, rows in rows_by_audio.items()
    }


def update_segment_text(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    text: str,
) -> dict[str, Any]:
    return _update_segment_fields(session, audio_file_id, segment_id, {"text": text})


def update_segment_phonemes(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    phon: str,
) -> dict[str, Any]:
    return _update_segment_fields(session, audio_file_id, segment_id, {"phon": phon})


def update_segment(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    payload: SegmentUpdate,
) -> dict[str, Any]:
    row = _find_segment(session, audio_file_id, segment_id)
    replacement = _segment_row(audio_file_id, row.position, payload, None, segment_id)
    session.delete(row)
    session.flush()
    session.add(replacement)
    _touch_audio(session, audio_file_id)
    session.commit()
    return replacement.as_payload()


def delete_segment(session: Session, audio_file_id: uuid.UUID, segment_id: str) -> None:
    row = _find_segment(session, audio_file_id, segment_id)
    session.delete(row)
    _touch_audio(session, audio_file_id)
    session.commit()


def _segment_row(
    audio_file_id: uuid.UUID,
    position: int,
    payload: SegmentPayload,
    fallback_accuracy: float | None,
    source_id: str | None = None,
) -> AudioSegment:
    raw = payload if isinstance(payload, dict) else payload.model_dump(mode="json")
    annotations = raw["annotations"]
    if "_source" in annotations["metadata"]:
        raise ValueError("segment annotation metadata key '_source' is reserved")
    accuracy = annotations["accuracy"]
    row = AudioSegment(
        audio_file_id=audio_file_id,
        source_id=source_id or str(raw.get("id") or uuid.uuid4()),
        position=position,
        start_seconds=float(raw["start"]),
        end_seconds=float(raw["end"]),
        text=str(raw["text"]),
        phon=str(raw["phon"]),
        kind=str(raw.get("type_", "segment")),
        accuracy=accuracy if accuracy is not None else fallback_accuracy,
        speaker_id=annotations["speaker_id"],
        metadata_={
            **annotations["metadata"],
            "_source": {
                "segment": {
                    key: value
                    for key, value in raw.items()
                    if key not in {"id", "start", "end", "text", "phon", "type_", "alignment", "annotations"}
                },
                "annotations": {
                    key: value
                    for key, value in annotations.items()
                    if key not in {"accuracy", "speaker_id", "metadata"}
                },
            },
        },
    )
    row.alignment = Alignment(data=raw.get("alignment"))
    return row


def _find_segment(session: Session, audio_file_id: uuid.UUID, segment_id: str) -> AudioSegment:
    row = session.execute(
        select(AudioSegment).where(
            AudioSegment.audio_file_id == audio_file_id,
            AudioSegment.source_id == segment_id,
        )
    ).unique().scalar_one_or_none()
    if row is None:
        raise KeyError(f"Segment not found: {segment_id}")
    return row


def _update_segment_fields(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    values: dict[str, str],
) -> dict[str, Any]:
    row = _find_segment(session, audio_file_id, segment_id)
    for field, value in values.items():
        setattr(row, field, value)
    _touch_audio(session, audio_file_id)
    session.commit()
    return row.as_payload()


def _touch_audio(session: Session, audio_file_id: uuid.UUID) -> None:
    session.execute(
        update(AudioFile)
        .where(AudioFile.id == audio_file_id)
        .values(
            updated_at=datetime.now(UTC),
            segment_count=select(func.count())
            .select_from(AudioSegment)
            .where(AudioSegment.audio_file_id == audio_file_id)
            .scalar_subquery(),
        )
    )
