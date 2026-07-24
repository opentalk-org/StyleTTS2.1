from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter, ObjectStore
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore


@dataclass(frozen=True)
class PruneResult:
    pruned_paths: list[str]
    moved_audio_files: int


def purge_orphaned_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
) -> list[str]:
    statement = (
        select(BucketFile)
        .where(~BucketFile.audio_files.any())
        .with_for_update()
    )
    packs = list(session.execute(statement).scalars().all())
    paths = [pack.path for pack in packs]
    for pack in packs:
        session.delete(pack)
    session.commit()
    resolved_store = _object_store(session, store)
    for path in paths:
        resolved_store.delete(path)
    return paths


def prune_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    prune_fragmented_audio_packs(
        session,
        _object_store(session, store),
        config,
    )


def prune_fragmented_audio_packs(
    session: Session,
    store: ObjectStore,
    config: AudioPackConfig = AudioPackConfig(),
) -> PruneResult:
    candidates = _fragmented_packs(session, config)
    if len(candidates) < 2:
        return PruneResult(pruned_paths=[], moved_audio_files=0)
    for pack in candidates:
        pack.sealed = True
    live_items = _prune_live_audio_files(session, candidates)
    pack_data = {
        pack.id: store.download(pack.path)
        for pack in candidates
    }
    writer = AudioPackWriter(session, store, config)
    for item in live_items:
        write = writer.append(_slice_from_pack_map(pack_data, item))
        item.bucket_file_id = write.bucket_file.id
        item.bucket_file = write.bucket_file
        item.byte_offset = write.byte_offset
        item.byte_length = write.byte_length
    writer.flush()
    pruned_paths = [pack.path for pack in candidates]
    for pack in candidates:
        session.delete(pack)
    session.commit()
    for path in pruned_paths:
        store.delete(path)
    return PruneResult(
        pruned_paths=pruned_paths,
        moved_audio_files=len(live_items),
    )


def _object_store(session: Session, store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(settings_crud.object_store_config(session))


def _fragmented_packs(
    session: Session,
    config: AudioPackConfig,
) -> list[BucketFile]:
    packs = session.execute(
        select(BucketFile).with_for_update()
    ).scalars().all()
    return [
        pack
        for pack in packs
        if pack.size == 0 or pack.used_bytes / pack.size < config.prune_used_ratio
    ]


def _prune_live_audio_files(
    session: Session,
    packs: list[BucketFile],
) -> list[AudioFile]:
    pack_ids = [pack.id for pack in packs]
    statement = (
        select(AudioFile)
        .where(AudioFile.bucket_file_id.in_(pack_ids))
        .with_for_update(of=AudioFile)
    )
    return list(session.execute(statement).scalars())


def _slice_from_pack_map(
    pack_data: dict[UUID, bytes],
    item: AudioFile,
) -> bytes:
    return _slice_from_pack(pack_data[item.bucket_file.id], item)


def _slice_from_pack(pack_data: bytes, item: AudioFile) -> bytes:
    start = item.byte_offset
    return pack_data[start:start + item.byte_length]
