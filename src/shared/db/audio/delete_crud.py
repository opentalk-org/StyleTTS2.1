from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import delete
from sqlalchemy.orm import Session

from shared.db.audio.pack_crud import bulk_delete_packed_audio_files
from shared.db.audio.pack_prune import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig, ObjectStore
from shared.db.datasets.models import dataset_audio_files
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.storage import S3ObjectStore


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
        delete(dataset_audio_files).where(dataset_audio_files.c.audio_file_id.in_(ids))
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
