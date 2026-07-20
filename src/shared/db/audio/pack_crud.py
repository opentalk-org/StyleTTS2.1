import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter, ObjectStore
from shared.db.audio.rows_crud import get_audio_files_bulk
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate


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


def read_packed_audio_file(session: Session, store: ObjectStore, audio_file_id: uuid.UUID) -> bytes:
    return bulk_read_packed_audio_files(session, store, [audio_file_id])[audio_file_id]


def read_packed_audio_part(
    session: Session,
    store: ObjectStore,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
) -> bytes:
    return bulk_read_packed_audio_parts(session, store, {audio_file_id: payload})[audio_file_id]


def bulk_read_packed_audio_files(
    session: Session,
    store: ObjectStore,
    audio_file_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, bytes]:
    ids = list(dict.fromkeys(audio_file_ids))
    items = list(get_audio_files_bulk(session, ids).values())
    for item in items:
        _assert_packed(item)
    pack_data = _download_packs(store, items)
    return {item.id: _slice_audio(pack_data, item) for item in items}


def bulk_read_packed_audio_parts(
    session: Session,
    store: ObjectStore,
    requests: dict[uuid.UUID, AudioPartRead],
) -> dict[uuid.UUID, bytes]:
    items = list(get_audio_files_bulk(session, list(requests)).values())
    for item in items:
        _assert_packed(item)
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


def delete_packed_audio_file(session: Session, audio_file_id: uuid.UUID) -> None:
    bulk_delete_packed_audio_files(session, [audio_file_id])


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
        voice_id=payload.annotations.voice_id,
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
    item.voice_id = payload.annotations.voice_id
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


def _download_packs(store: ObjectStore, items: Sequence[AudioFile]) -> dict[uuid.UUID, bytes]:
    for item in items:
        _assert_packed(item)
        assert item.bucket_file is not None
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


def _assert_packed(item: AudioFile) -> None:
    if item.storage_kind != "packed":
        raise ValueError(f"Audio {item.id} contains metadata only; no stored audio bytes are available")
