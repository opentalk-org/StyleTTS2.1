import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from shared.db.audio.catalog import get_audio_files_bulk
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
)
from shared.db.audio.pack_prune import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig, ObjectStore
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate
from shared.db.datasets.models import dataset_audio_files
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.storage import S3ObjectStore


def create_audio_file(
    session: Session,
    payload: AudioCreate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return create_packed_audio_file(
        session,
        _object_store(session, store),
        payload,
        config,
    )


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> list[AudioFile]:
    return bulk_create_packed_audio_files(
        session,
        _object_store(session, store),
        payloads,
        config,
        commit=commit,
    )


def read_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_file(
        session,
        _object_store(session, store),
        audio_file_id,
    )


def read_audio_part(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_part(
        session,
        _object_store(session, store),
        audio_file_id,
        payload,
    )


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_files(
        session,
        _object_store(session, store),
        audio_file_ids,
    )


def bulk_read_audio_parts(
    session: Session,
    requests: dict[uuid.UUID, AudioPartRead],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_parts(
        session,
        _object_store(session, store),
        requests,
    )


def update_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioUpdate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return bulk_update_audio_files(
        session,
        {audio_file_id: payload},
        store=store,
        config=config,
    )[audio_file_id]


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
    resolved_store = None
    if binary_payloads:
        resolved_store = _object_store(session, store)
        items.update(
            bulk_update_packed_audio_files(
                session,
                resolved_store,
                binary_payloads,
                config,
                commit=False,
            )
        )
        waveform_crud.bulk_delete_waveforms(
            session,
            list(binary_payloads),
            commit=False,
        )
    if metadata_payloads:
        metadata_items = get_audio_files_bulk(session, list(metadata_payloads))
        for audio_file_id, payload in metadata_payloads.items():
            item = metadata_items[audio_file_id]
            _update_audio_metadata(item, payload)
            items[audio_file_id] = item
    if items:
        session.commit()
    if resolved_store is not None:
        prune_fragmented_audio_packs(session, resolved_store, config)
    return items


def delete_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    bulk_delete_audio_files(session, [audio_file_id], store=store, config=config)


def bulk_delete_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
    prune: bool = False,
) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    resolved_store = _object_store(session, store)
    waveform_crud.bulk_delete_waveforms(session, ids, commit=False)
    session.execute(
        delete(dataset_audio_files).where(
            dataset_audio_files.c.audio_file_id.in_(ids)
        )
    )
    bulk_delete_packed_audio_files(session, ids, commit=False)
    if commit:
        session.commit()
    if prune:
        assert commit, "audio pack pruning requires committed deletes"
        prune_fragmented_audio_packs(session, resolved_store, config)


def _object_store(session: Session, store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(settings_crud.object_store_config(session))


def _update_audio_metadata(item: AudioFile, payload: AudioUpdate) -> None:
    item.name = payload.name
    item.duration = payload.duration
    item.speaker_id = payload.annotations.speaker_id
    item.score = payload.annotations.score
    item.accuracy = payload.annotations.accuracy
    if "language" in payload.model_fields_set:
        item.language = payload.language
    if "style_prompt" in payload.model_fields_set:
        item.style_prompt = payload.style_prompt
    if "voice_prompt" in payload.model_fields_set:
        item.voice_prompt = payload.voice_prompt
    item.segments = payload.segments
    item.metadata_ = payload.annotations.metadata
    item.virtual = payload.virtual
    item.updated_at = datetime.now(UTC)
