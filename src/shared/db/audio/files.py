import uuid
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from shared.db.audio.catalog import get_audio_files_bulk
from shared.db.audio.models import AudioFile
from shared.db.audio.packed import (
    bulk_create_packed_audio_files,
    bulk_delete_packed_audio_files,
    bulk_update_packed_audio_files,
    create_packed_audio_file,
)
from shared.db.audio.maintenance import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate
from shared.db.audio.segments import bulk_replace_audio_segments
from shared.db.audio.storage_locations import audio_storage_locations
from shared.db.datasets.models import dataset_audio_files
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.storage import ObjectRange, S3RequestMetrics


def create_audio_file(
    session: Session,
    payload: AudioCreate,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return create_packed_audio_file(
        session,
        settings_crud.object_store(session),
        payload,
        config,
    )


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> list[AudioFile]:
    return bulk_create_packed_audio_files(
        session,
        settings_crud.object_store(session),
        payloads,
        config,
        commit=commit,
    )


def read_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
) -> bytes:
    location = audio_storage_locations(session, [audio_file_id])[audio_file_id]
    return settings_crud.object_store(session).read_range(
        ObjectRange(
            location.object_path,
            location.byte_offset,
            location.byte_length,
        )
    )


def read_audio_part(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
) -> bytes:
    location = audio_storage_locations(session, [audio_file_id])[audio_file_id]
    if payload.start < 0:
        raise ValueError(f"part start must be non-negative: {payload.start}")
    if payload.length <= 0:
        raise ValueError(f"part length must be positive: {payload.length}")
    if payload.start + payload.length > location.byte_length:
        raise ValueError(f"part exceeds audio length: {audio_file_id}")
    return settings_crud.object_store(session).read_range(
        ObjectRange(
            location.object_path,
            location.byte_offset + payload.start,
            payload.length,
        )
    )


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    request_metrics: S3RequestMetrics | None = None,
) -> dict[uuid.UUID, bytes]:
    ids = list(dict.fromkeys(audio_file_ids))
    locations = audio_storage_locations(session, ids)
    ranges = [
        ObjectRange(
            locations[audio_file_id].object_path,
            locations[audio_file_id].byte_offset,
            locations[audio_file_id].byte_length,
        )
        for audio_file_id in ids
    ]
    store = settings_crud.object_store(session, request_metrics)
    started = time.monotonic()
    payloads = store.read_ranges(ranges)
    if request_metrics is not None:
        request_metrics.fetch_seconds = time.monotonic() - started
        request_metrics.fetch_bytes = sum(len(payload) for payload in payloads)
    return dict(zip(ids, payloads, strict=True))


def update_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioUpdate,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return bulk_update_audio_files(
        session,
        {audio_file_id: payload},
        config=config,
    )[audio_file_id]


def bulk_update_audio_files(
    session: Session,
    payloads: dict[uuid.UUID, AudioUpdate],
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
        store = settings_crud.object_store(session)
        items.update(
            bulk_update_packed_audio_files(
                session,
                store,
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
        bulk_replace_audio_segments(
            session,
            {
                audio_file_id: payload.segments
                for audio_file_id, payload in metadata_payloads.items()
            },
            commit=False,
            fallback_accuracy={
                audio_file_id: payload.annotations.accuracy
                for audio_file_id, payload in metadata_payloads.items()
            },
        )
    if items:
        session.commit()
    if binary_payloads:
        prune_fragmented_audio_packs(session, store, config)
    return items


def delete_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    bulk_delete_audio_files(session, [audio_file_id], config=config)


def bulk_delete_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
    prune: bool = False,
) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    store = settings_crud.object_store(session)
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
        prune_fragmented_audio_packs(session, store, config)

def _update_audio_metadata(item: AudioFile, payload: AudioUpdate) -> None:
    item.name = payload.name
    item.duration = payload.duration
    item.speaker_id = payload.annotations.speaker_id
    item.score = payload.annotations.score
    if "language" in payload.model_fields_set:
        item.language = payload.language
    if "style_prompt" in payload.model_fields_set:
        item.style_prompt = payload.style_prompt
    if "voice_prompt" in payload.model_fields_set:
        item.voice_prompt = payload.voice_prompt
    item.metadata_ = payload.annotations.metadata
    item.virtual = payload.virtual
    item.updated_at = datetime.now(UTC)
