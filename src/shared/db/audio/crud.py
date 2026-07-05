import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.pack_crud import (
    bulk_create_packed_audio_files,
    bulk_delete_packed_audio_files,
    bulk_read_packed_audio_files,
    bulk_read_packed_audio_parts,
    bulk_update_packed_audio_files,
    create_packed_audio_file,
    read_packed_audio_file,
    read_packed_audio_part,
    update_packed_audio_file,
)
from shared.db.audio.pack_prune import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig, ObjectStore
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate, SegmentCreate, SegmentUpdate
from shared.db.common import many, one
from shared.storage import ObjectStoreConfig, S3ObjectStore


def list_audio_files(session: Session) -> Sequence[AudioFile]:
    return many(session, AudioFile)


def get_audio_file(session: Session, audio_file_id: uuid.UUID) -> AudioFile:
    return one(session, AudioFile, audio_file_id)


def create_audio_file(
    session: Session,
    payload: AudioCreate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return create_packed_audio_file(session, _object_store(store), payload, config)


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> list[AudioFile]:
    return bulk_create_packed_audio_files(session, _object_store(store), payloads, config)


def read_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_file(session, _object_store(store), audio_file_id)


def read_audio_part(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_part(session, _object_store(store), audio_file_id, payload)


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_files(session, _object_store(store), audio_file_ids)


def bulk_read_audio_parts(
    session: Session,
    requests: dict[uuid.UUID, AudioPartRead],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_parts(session, _object_store(store), requests)


def update_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioUpdate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    if payload.wav_bytes is None:
        item = one(session, AudioFile, audio_file_id)
        _update_audio_metadata(item, payload)
        session.commit()
        session.refresh(item)
        return item
    resolved_store = _object_store(store)
    item = update_packed_audio_file(session, resolved_store, audio_file_id, payload, config)
    prune_fragmented_audio_packs(session, resolved_store, config)
    session.refresh(item)
    return item


def bulk_update_audio_files(
    session: Session,
    payloads: dict[uuid.UUID, AudioUpdate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> dict[uuid.UUID, AudioFile]:
    binary_payloads = {
        audio_file_id: payload
        for audio_file_id, payload in payloads.items()
        if payload.wav_bytes is not None
    }
    metadata_payloads = {
        audio_file_id: payload
        for audio_file_id, payload in payloads.items()
        if payload.wav_bytes is None
    }
    items: dict[uuid.UUID, AudioFile] = {}
    if binary_payloads:
        resolved_store = _object_store(store)
        items.update(bulk_update_packed_audio_files(session, resolved_store, binary_payloads, config))
        prune_fragmented_audio_packs(session, resolved_store, config)
    if metadata_payloads:
        for audio_file_id, payload in metadata_payloads.items():
            item = one(session, AudioFile, audio_file_id)
            _update_audio_metadata(item, payload)
            items[audio_file_id] = item
        session.commit()
    for item in items.values():
        session.refresh(item)
    return items


def delete_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    resolved_store = _object_store(store)
    bulk_delete_packed_audio_files(session, [audio_file_id])
    prune_fragmented_audio_packs(session, resolved_store, config)


def bulk_delete_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    resolved_store = _object_store(store)
    bulk_delete_packed_audio_files(session, audio_file_ids)
    prune_fragmented_audio_packs(session, resolved_store, config)


def create_segment(session: Session, audio_file_id: uuid.UUID, payload: SegmentCreate) -> dict[str, Any]:
    item = one(session, AudioFile, audio_file_id)
    segment = {"id": str(uuid.uuid4()), **payload.model_dump(mode="json")}
    item.segments = [*item.segments, segment]
    session.commit()
    return segment


def update_segment(
    session: Session,
    audio_file_id: uuid.UUID,
    segment_id: str,
    payload: SegmentUpdate,
) -> dict[str, Any]:
    item = one(session, AudioFile, audio_file_id)
    replacement = {"id": segment_id, **payload.model_dump(mode="json")}
    segments = [replacement if segment["id"] == segment_id else segment for segment in item.segments]
    if segments == item.segments:
        raise KeyError(f"Segment not found: {segment_id}")
    item.segments = segments
    session.commit()
    return replacement


def delete_segment(session: Session, audio_file_id: uuid.UUID, segment_id: str) -> None:
    item = one(session, AudioFile, audio_file_id)
    segments = [segment for segment in item.segments if segment["id"] != segment_id]
    if len(segments) == len(item.segments):
        raise KeyError(f"Segment not found: {segment_id}")
    item.segments = segments
    session.commit()


def _object_store(store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(ObjectStoreConfig.from_env())


def _update_audio_metadata(item: AudioFile, payload: AudioUpdate) -> None:
    item.name = payload.name
    item.duration = payload.duration
    item.segments = payload.segments
    item.metadata_ = payload.metadata
    item.virtual = payload.virtual
