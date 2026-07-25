import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter
from shared.db.audio.catalog import get_audio_files_bulk
from shared.db.audio.schemas import AudioCreate, AudioUpdate
from shared.storage import ObjectStore


def create_packed_audio_file(
    session: Session,
    store: ObjectStore,
    payload: AudioCreate,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    item = bulk_create_packed_audio_files(session, store, [payload], config)[0]
    return item


def bulk_create_packed_audio_files(
    session: Session,
    store: ObjectStore,
    payloads: Sequence[AudioCreate],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> list[AudioFile]:
    writer = AudioPackWriter(session, store, config)
    items = [_create_item_from_write(writer, payload) for payload in payloads]
    writer.flush()
    session.add_all(items)
    session.flush()
    if commit:
        session.commit()
    return items


def bulk_update_packed_audio_files(
    session: Session,
    store: ObjectStore,
    payloads: dict[uuid.UUID, AudioUpdate],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> dict[uuid.UUID, AudioFile]:
    writer = AudioPackWriter(session, store, config)
    items = get_audio_files_bulk(session, list(payloads))
    for audio_file_id, item in items.items():
        _decrease_used_bytes(item)
        _replace_item_from_write(item, writer, payloads[audio_file_id])
    writer.flush()
    if commit:
        session.commit()
    return items

def bulk_delete_packed_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    commit: bool = True,
) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    rows = session.execute(
        select(
            AudioFile.id,
            AudioFile.bucket_file_id,
            AudioFile.byte_length,
            AudioFile.storage_kind,
        ).where(AudioFile.id.in_(ids))
    ).all()
    loaded_ids = {row.id for row in rows}
    missing_ids = set(ids).difference(loaded_ids)
    if missing_ids:
        missing = sorted(str(audio_file_id) for audio_file_id in missing_ids)
        raise KeyError(f"Audio files not found: {missing}")
    removed_by_pack: dict[uuid.UUID, int] = {}
    for row in rows:
        if row.storage_kind == "external":
            assert row.bucket_file_id is None, f"external audio has a bucket: {row.id}"
            continue
        assert row.bucket_file_id is not None, f"packed audio has no bucket: {row.id}"
        removed_by_pack[row.bucket_file_id] = removed_by_pack.get(row.bucket_file_id, 0) + row.byte_length
    for pack_id, removed_bytes in removed_by_pack.items():
        result = session.execute(
            update(BucketFile)
            .where(BucketFile.id == pack_id, BucketFile.used_bytes >= removed_bytes)
            .values(used_bytes=BucketFile.used_bytes - removed_bytes)
        )
        assert result.rowcount == 1, f"audio pack used bytes would go negative: {pack_id}"
    session.execute(delete(AudioFile).where(AudioFile.id.in_(ids)))
    if commit:
        session.commit()


def _create_item_from_write(writer: AudioPackWriter, payload: AudioCreate) -> AudioFile:
    write = writer.append(payload.wav_bytes)
    return AudioFile(
        name=payload.name,
        bucket_file_id=write.bucket_file.id,
        byte_offset=write.byte_offset,
        byte_length=write.byte_length,
        duration=payload.duration,
        speaker_id=payload.annotations.speaker_id,
        score=payload.annotations.score,
        accuracy=payload.annotations.accuracy,
        language=payload.language,
        style_prompt=payload.style_prompt,
        voice_prompt=payload.voice_prompt,
        segments=payload.segments,
        metadata_=payload.annotations.metadata,
        virtual=payload.virtual,
        storage_kind="packed",
        storage_ref=None,
        updated_at=_now(),
    )


def _replace_item_from_write(item: AudioFile, writer: AudioPackWriter, payload: AudioUpdate) -> None:
    assert payload.wav_bytes is not None, f"audio bytes are required for packed update: {item.id}"
    write = writer.append(payload.wav_bytes)
    item.name = payload.name
    item.bucket_file_id = write.bucket_file.id
    item.bucket_file = write.bucket_file
    item.byte_offset = write.byte_offset
    item.byte_length = write.byte_length
    item.duration = payload.duration
    item.speaker_id = payload.annotations.speaker_id
    item.score = payload.annotations.score
    item.accuracy = payload.annotations.accuracy
    item.segments = payload.segments
    item.metadata_ = payload.annotations.metadata
    item.virtual = payload.virtual
    item.storage_kind = "packed"
    item.storage_ref = None
    item.updated_at = _now()


def _decrease_used_bytes(item: AudioFile) -> None:
    if item.storage_kind == "external":
        assert item.bucket_file is None, f"external audio has a bucket: {item.id}"
        return
    _assert_packed(item)
    assert item.bucket_file is not None
    item.bucket_file.used_bytes -= item.byte_length
    assert item.bucket_file.used_bytes >= 0, f"pack used bytes went negative: {item.bucket_file_id}"


def _now() -> datetime:
    return datetime.now(UTC)


def _assert_packed(item: AudioFile) -> None:
    if item.storage_kind != "packed":
        raise ValueError(f"Audio {item.id} contains metadata only; no stored audio bytes are available")
