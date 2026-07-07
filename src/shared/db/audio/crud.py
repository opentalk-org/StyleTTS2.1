import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Text, cast, desc, func, or_, select
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
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate
from shared.db.audio.segments_crud import (
    create_segment,
    delete_segment,
    list_audio_segments,
    replace_audio_segments,
    bulk_replace_audio_segments,
    update_segment,
    update_segment_phonemes,
    update_segment_text,
)
from shared.db.common import many, one
from shared.db.settings import crud as settings_crud
from shared.db.datasets.models import Dataset
from shared.db.waveforms import crud as waveform_crud
from shared.storage import S3ObjectStore


def list_audio_files(session: Session) -> Sequence[AudioFile]:
    return many(session, AudioFile)


def list_audio_files_by_run(session: Session, run_id: str) -> Sequence[AudioFile]:
    statement = (
        select(AudioFile)
        .where(AudioFile.metadata_["run_id"].astext == run_id)
        .order_by(AudioFile.updated_at.asc())
    )
    return session.execute(statement).unique().scalars().all()


def search_audio_files(
    session: Session,
    query: str,
    dataset: str,
    sort: str,
    limit: int,
    offset: int,
) -> tuple[Sequence[AudioFile], int]:
    statement = select(AudioFile)
    count_statement = select(func.count()).select_from(AudioFile)
    for item in _audio_filters(query, dataset):
        statement = statement.where(item)
        count_statement = count_statement.where(item)
    statement = statement.order_by(_audio_sort(sort)).limit(limit).offset(offset)
    rows = session.execute(statement).unique().scalars().all()
    total = session.execute(count_statement).scalar_one()
    return rows, total


def search_audio_file_ids(session: Session, query: str, dataset: str) -> list[uuid.UUID]:
    statement = select(AudioFile.id)
    for item in _audio_filters(query, dataset):
        statement = statement.where(item)
    return list(session.execute(statement).scalars().all())


def get_audio_file(session: Session, audio_file_id: uuid.UUID) -> AudioFile:
    return one(session, AudioFile, audio_file_id)


def _audio_filters(query: str, dataset: str) -> list[Any]:
    filters = []
    if query:
        pattern = f"%{query}%"
        filters.append(or_(AudioFile.name.ilike(pattern), cast(AudioFile.metadata_, Text).ilike(pattern)))
    if dataset == "unassigned":
        filters.append(~AudioFile.datasets.any())
    elif dataset != "all":
        dataset_id = uuid.UUID(dataset)
        filters.append(AudioFile.datasets.any(Dataset.id == dataset_id))
    return filters


def _audio_sort(sort: str):
    if sort == "name":
        return AudioFile.name
    if sort == "duration":
        return desc(AudioFile.duration)
    if sort == "segments":
        return desc(func.jsonb_array_length(AudioFile.segments))
    if sort == "speaker":
        return AudioFile.name
    return desc(AudioFile.updated_at)


def create_audio_file(
    session: Session,
    payload: AudioCreate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
    create_waveform: bool = True,
) -> AudioFile:
    resolved_store = _object_store(session, store)
    item = create_packed_audio_file(session, resolved_store, payload, config)
    if create_waveform:
        waveform_crud.replace_waveform_from_audio(session, item.id, payload.wav_bytes, item.duration, payload.waveform, resolved_store)
    return item


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
    create_waveforms: bool = True,
) -> list[AudioFile]:
    resolved_store = _object_store(session, store)
    items = bulk_create_packed_audio_files(session, resolved_store, payloads, config)
    if create_waveforms:
        waveform_crud.bulk_replace_waveforms_from_audio(
            session,
            [(item.id, payload.wav_bytes, item.duration, payload.waveform) for item, payload in zip(items, payloads, strict=True)],
            resolved_store,
        )
    return items


def read_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_file(session, _object_store(session, store), audio_file_id)


def read_audio_part(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_part(session, _object_store(session, store), audio_file_id, payload)


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_files(session, _object_store(session, store), audio_file_ids)


def bulk_read_audio_parts(
    session: Session,
    requests: dict[uuid.UUID, AudioPartRead],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_parts(session, _object_store(session, store), requests)


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
    resolved_store = _object_store(session, store)
    item = update_packed_audio_file(session, resolved_store, audio_file_id, payload, config)
    waveform_crud.replace_waveform_from_audio(session, item.id, payload.wav_bytes, item.duration, payload.waveform, resolved_store)
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
        resolved_store = _object_store(session, store)
        items.update(bulk_update_packed_audio_files(session, resolved_store, binary_payloads, config))
        for audio_file_id, payload in binary_payloads.items():
            item = items[audio_file_id]
            waveform_crud.replace_waveform_from_audio(session, item.id, payload.wav_bytes, item.duration, payload.waveform, resolved_store)
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
    resolved_store = _object_store(session, store)
    waveform_crud.delete_waveform(session, audio_file_id)
    bulk_delete_packed_audio_files(session, [audio_file_id])
    prune_fragmented_audio_packs(session, resolved_store, config)


def bulk_delete_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    resolved_store = _object_store(session, store)
    ids = list(audio_file_ids)
    for audio_file_id in ids:
        waveform_crud.delete_waveform(session, audio_file_id)
    bulk_delete_packed_audio_files(session, ids)
    prune_fragmented_audio_packs(session, resolved_store, config)


def _object_store(session: Session, store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(settings_crud.object_store_config(session))


def _update_audio_metadata(item: AudioFile, payload: AudioUpdate) -> None:
    item.name = payload.name
    item.duration = payload.duration
    if "score" in payload.model_fields_set:
        item.score = payload.score
    item.segments = payload.segments
    item.metadata_ = payload.metadata
    item.virtual = payload.virtual
    item.updated_at = _now()


def _now() -> datetime:
    return datetime.now(UTC)
