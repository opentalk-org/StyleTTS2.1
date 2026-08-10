from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import bindparam, delete, select, update
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter
from shared.db.audio.schemas import AudioRecoveryPack
from shared.db.pack_folders import PackFolderAllocator
from shared.db.settings import crud as settings_crud
from shared.db.staged_objects import register_staged_object
from shared.storage import ObjectStore


@dataclass(frozen=True)
class PruneResult:
    pruned_paths: list[str]
    moved_audio_files: int


@dataclass(frozen=True)
class CompactResult:
    source_packs: int
    source_bytes: int
    live_bytes: int
    moved_audio_files: int
    replacement_packs: int


@dataclass(frozen=True)
class RecoveryResult:
    moved_audio_files: int
    removed_missing_packs: int
    replacement_paths: list[str]


def recover_missing_audio_packs(
    session: Session,
    missing_bucket_ids: Sequence[UUID],
    recovered_packs: Sequence[AudioRecoveryPack],
    config: AudioPackConfig = AudioPackConfig(),
) -> RecoveryResult:
    store = settings_crud.object_store(session)
    missing_ids = set(missing_bucket_ids)
    source_packs = list(session.execute(
        select(BucketFile).where(BucketFile.id.in_(missing_ids)).with_for_update()
    ).scalars())
    assert {pack.id for pack in source_packs} == missing_ids
    assert all(not store.exists(pack.path) for pack in source_packs)
    source_rows = session.execute(
        select(AudioFile.id, AudioFile.bucket_file_id, AudioFile.byte_length)
        .where(AudioFile.bucket_file_id.in_(missing_ids)).with_for_update()
    ).all()
    source = {row.id: row for row in source_rows}
    placements = [item for pack in recovered_packs for item in pack.placements]
    assert len(placements) == len(source) == len({item.audio_file_id for item in placements})
    assert {item.audio_file_id for item in placements} == set(source)
    allocator = PackFolderAllocator(session, BucketFile, config.path_prefix, config.folder_target_files)
    replacement_paths = []
    update_rows = []
    for recovered in recovered_packs:
        size = recovered.local_path.stat().st_size
        cursor = 0
        for item in sorted(recovered.placements, key=lambda value: value.byte_offset):
            assert item.byte_offset == cursor
            assert item.byte_length == source[item.audio_file_id].byte_length
            cursor += item.byte_length
        assert cursor == size and 0 < size <= config.target_pack_bytes
        replacement_id = uuid4()
        replacement = BucketFile(
            id=replacement_id,
            path=allocator.path_for(replacement_id),
            size=size,
            used_bytes=size,
            sealed=True,
        )
        session.add(replacement)
        session.flush()
        register_staged_object(session, store, replacement.path)
        store.upload_path(replacement.path, recovered.local_path)
        assert store.exists(replacement.path)
        replacement_paths.append(replacement.path)
        update_rows.extend({"audio_id": item.audio_file_id, "bucket_id": replacement.id,
                            "byte_offset": item.byte_offset} for item in recovered.placements)
    statement = update(AudioFile.__table__).where(
        AudioFile.__table__.c.id == bindparam("audio_id")
    ).values(bucket_file_id=bindparam("bucket_id"), byte_offset=bindparam("byte_offset"))
    session.execute(statement, update_rows)
    session.execute(delete(BucketFile).where(BucketFile.id.in_(missing_ids)))
    session.commit()
    return RecoveryResult(len(source), len(source_packs), replacement_paths)


def purge_orphaned_audio_packs(
    session: Session,
) -> list[str]:
    store = settings_crud.object_store(session)
    removed_paths: list[str] = []
    while paths := purge_orphaned_audio_pack_batch(session, store):
        removed_paths.extend(paths)
    return removed_paths


def purge_orphaned_audio_pack_batch(
    session: Session,
    store: ObjectStore,
    batch_size: int = 256,
    workers: int = 9,
) -> list[str]:
    statement = (
        select(BucketFile)
        .where(~BucketFile.audio_files.any())
        .order_by(BucketFile.id)
        .limit(batch_size)
        .with_for_update()
    )
    packs = list(session.execute(statement).scalars().all())
    paths = [pack.path for pack in packs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(store.delete, paths))
    for pack in packs:
        session.delete(pack)
    session.commit()
    return paths


def compact_audio_pack_batch(
    session: Session,
    store: ObjectStore,
    config: AudioPackConfig = AudioPackConfig(),
    max_source_bytes: int = 512 * 1024 * 1024,
) -> CompactResult:
    """Rewrite one bounded batch while retaining source packs as DB-tracked orphans."""
    candidates = _prunable_packs(session, config)
    compact = _bounded_pack_batch(candidates, max_source_bytes)
    if len(compact) < 2:
        session.rollback()
        return CompactResult(0, 0, 0, 0, 0)

    source_bytes = sum(pack.size for pack in compact)
    live_bytes = sum(pack.used_bytes for pack in compact)
    live_items = _prune_live_audio_files(session, compact)
    pack_data = _download_pack_data(store, compact, config.remote_workers)
    _validate_pack_data(compact, pack_data)

    writer = AudioPackWriter(session, store, config)
    replacement_ids: set[UUID] = set()
    for item in live_items:
        audio_bytes = _slice_from_pack_map(pack_data, item)
        if len(audio_bytes) != item.byte_length:
            raise EOFError(f"audio range exceeds source pack: {item.id}")
        write = writer.append(audio_bytes)
        replacement_ids.add(write.bucket_file.id)
        item.bucket_file_id = write.bucket_file.id
        item.bucket_file = write.bucket_file
        item.byte_offset = write.byte_offset
        item.byte_length = write.byte_length
    writer.flush(verify=True)
    session.commit()
    return CompactResult(
        source_packs=len(compact),
        source_bytes=source_bytes,
        live_bytes=live_bytes,
        moved_audio_files=len(live_items),
        replacement_packs=len(replacement_ids),
    )


def prune_audio_packs(
    session: Session,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    prune_fragmented_audio_packs(
        session,
        settings_crud.object_store(session),
        config,
    )


def prune_fragmented_audio_packs(
    session: Session,
    store: ObjectStore,
    config: AudioPackConfig = AudioPackConfig(),
) -> PruneResult:
    candidates = _prunable_packs(session, config)
    empty = [pack for pack in candidates if pack.used_bytes == 0]
    compact = [pack for pack in candidates if pack.used_bytes > 0]
    if len(compact) < 2 and not empty:
        return PruneResult(pruned_paths=[], moved_audio_files=0)
    for pack in candidates:
        pack.sealed = True
    live_items = _prune_live_audio_files(session, compact) if len(compact) >= 2 else []
    if live_items:
        pack_data = {pack.id: store.download(pack.path) for pack in compact}
        writer = AudioPackWriter(session, store, config)
        for item in live_items:
            write = writer.append(_slice_from_pack_map(pack_data, item))
            item.bucket_file_id = write.bucket_file.id
            item.bucket_file = write.bucket_file
            item.byte_offset = write.byte_offset
            item.byte_length = write.byte_length
        writer.flush()
    removed = empty + (compact if len(compact) >= 2 else [])
    pruned_paths = [pack.path for pack in removed]
    for pack in removed:
        session.delete(pack)
    session.commit()
    for path in pruned_paths:
        store.delete(path)
    return PruneResult(
        pruned_paths=pruned_paths,
        moved_audio_files=len(live_items),
    )

def _prunable_packs(
    session: Session,
    config: AudioPackConfig,
) -> list[BucketFile]:
    packs = session.execute(
        select(BucketFile)
        .where(BucketFile.audio_files.any())
        .order_by(BucketFile.id)
        .with_for_update()
    ).scalars().all()
    return [
        pack
        for pack in packs
        if (
            pack.size < config.target_pack_bytes * config.prune_size_ratio
            or pack.used_bytes / pack.size < config.prune_used_ratio
        )
    ]


def _bounded_pack_batch(
    candidates: list[BucketFile],
    max_source_bytes: int,
) -> list[BucketFile]:
    selected: list[BucketFile] = []
    selected_bytes = 0
    for pack in candidates:
        if selected and selected_bytes + pack.size > max_source_bytes:
            break
        selected.append(pack)
        selected_bytes += pack.size
    return selected


def _validate_pack_data(
    packs: list[BucketFile],
    pack_data: dict[UUID, bytes],
) -> None:
    for pack in packs:
        actual_size = len(pack_data[pack.id])
        if actual_size != pack.size:
            raise IOError(
                f"source audio pack size mismatch for {pack.path}: "
                f"expected {pack.size}, got {actual_size}"
            )


def _download_pack_data(
    store: ObjectStore,
    packs: list[BucketFile],
    workers: int,
) -> dict[UUID, bytes]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        payloads = executor.map(store.download, [pack.path for pack in packs])
        return {pack.id: payload for pack, payload in zip(packs, payloads, strict=True)}


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
