import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter, ObjectStore
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate
from shared.db.common import one


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
) -> list[AudioFile]:
    writer = AudioPackWriter(session, store, config)
    items = [_create_item_from_write(writer, payload) for payload in payloads]
    writer.flush()
    session.add_all(items)
    session.commit()
    for item in items:
        session.refresh(item)
    return items


def read_packed_audio_file(session: Session, store: ObjectStore, audio_file_id: uuid.UUID) -> bytes:
    item = one(session, AudioFile, audio_file_id)
    return store.read_range(item.bucket_file.path, item.byte_offset, item.byte_length)


def read_packed_audio_part(
    session: Session,
    store: ObjectStore,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
) -> bytes:
    item = one(session, AudioFile, audio_file_id)
    _assert_valid_part(item, payload)
    return store.read_range(item.bucket_file.path, item.byte_offset + payload.start, payload.length)


def bulk_read_packed_audio_files(
    session: Session,
    store: ObjectStore,
    audio_file_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, bytes]:
    items = [one(session, AudioFile, audio_file_id) for audio_file_id in audio_file_ids]
    pack_data = _download_packs(store, items)
    return {item.id: _slice_audio(pack_data, item) for item in items}


def bulk_read_packed_audio_parts(
    session: Session,
    store: ObjectStore,
    requests: dict[uuid.UUID, AudioPartRead],
) -> dict[uuid.UUID, bytes]:
    items = [one(session, AudioFile, audio_file_id) for audio_file_id in requests]
    for item in items:
        _assert_valid_part(item, requests[item.id])
    pack_data = _download_packs(store, items)
    return {
        item.id: _slice_audio_part(pack_data, item, requests[item.id])
        for item in items
    }


def update_packed_audio_file(
    session: Session,
    store: ObjectStore,
    audio_file_id: uuid.UUID,
    payload: AudioUpdate,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    items = bulk_update_packed_audio_files(session, store, {audio_file_id: payload}, config)
    return items[audio_file_id]


def bulk_update_packed_audio_files(
    session: Session,
    store: ObjectStore,
    payloads: dict[uuid.UUID, AudioUpdate],
    config: AudioPackConfig = AudioPackConfig(),
) -> dict[uuid.UUID, AudioFile]:
    writer = AudioPackWriter(session, store, config)
    items = {audio_file_id: one(session, AudioFile, audio_file_id) for audio_file_id in payloads}
    for audio_file_id, item in items.items():
        _decrease_used_bytes(item)
        _replace_item_from_write(item, writer, payloads[audio_file_id])
    writer.flush()
    session.commit()
    for item in items.values():
        session.refresh(item)
    return items


def delete_packed_audio_file(session: Session, audio_file_id: uuid.UUID) -> None:
    bulk_delete_packed_audio_files(session, [audio_file_id])


def bulk_delete_packed_audio_files(session: Session, audio_file_ids: Iterable[uuid.UUID]) -> None:
    items = [one(session, AudioFile, audio_file_id) for audio_file_id in audio_file_ids]
    for item in items:
        _decrease_used_bytes(item)
        session.delete(item)
    session.commit()


def _create_item_from_write(writer: AudioPackWriter, payload: AudioCreate) -> AudioFile:
    write = writer.append(payload.wav_bytes)
    return AudioFile(
        name=payload.name,
        bucket_file_id=write.bucket_file.id,
        byte_offset=write.byte_offset,
        byte_length=write.byte_length,
        duration=payload.duration,
        score=payload.score,
        language=payload.language,
        style_prompt=payload.style_prompt,
        voice_prompt=payload.voice_prompt,
        segments=payload.segments,
        metadata_=payload.metadata,
        virtual=payload.virtual,
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
    if "score" in payload.model_fields_set:
        item.score = payload.score
    item.segments = payload.segments
    item.metadata_ = payload.metadata
    item.virtual = payload.virtual
    item.updated_at = _now()


def _decrease_used_bytes(item: AudioFile) -> None:
    item.bucket_file.used_bytes -= item.byte_length
    assert item.bucket_file.used_bytes >= 0, f"pack used bytes went negative: {item.bucket_file_id}"


def _now() -> datetime:
    return datetime.now(UTC)


def _download_packs(store: ObjectStore, items: Sequence[AudioFile]) -> dict[uuid.UUID, bytes]:
    packs = {item.bucket_file.id: item.bucket_file for item in items}
    return {pack_id: store.download(pack.path) for pack_id, pack in packs.items()}


def _slice_audio(pack_data: dict[uuid.UUID, bytes], item: AudioFile) -> bytes:
    start = item.byte_offset
    end = start + item.byte_length
    return pack_data[item.bucket_file.id][start:end]


def _slice_audio_part(pack_data: dict[uuid.UUID, bytes], item: AudioFile, payload: AudioPartRead) -> bytes:
    start = item.byte_offset + payload.start
    end = start + payload.length
    return pack_data[item.bucket_file.id][start:end]


def _assert_valid_part(item: AudioFile, payload: AudioPartRead) -> None:
    assert payload.start >= 0, f"part start must be non-negative: {payload.start}"
    assert payload.length > 0, f"part length must be positive: {payload.length}"
    assert payload.start + payload.length <= item.byte_length, f"part exceeds audio length: {item.id}"
