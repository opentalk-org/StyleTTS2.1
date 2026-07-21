from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter, ObjectStore
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore


@dataclass(frozen=True)
class RepackResult:
    moved_audio_files: int
    replaced_packs: int
    created_packs: int
    bytes_verified: int
    deleted_paths: tuple[str, ...]
    remaining_packs: int


def repack_legacy_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
    max_source_bytes: int = 512 * 1024 * 1024,
    config: AudioPackConfig = AudioPackConfig(),
) -> RepackResult:
    if max_source_bytes < 1:
        raise ValueError("repack source-byte limit must be positive")
    resolved_store = store or S3ObjectStore(settings_crud.object_store_config(session))
    legacy = _legacy_packs(session, config.path_prefix)
    selected = _bounded_packs(legacy, max_source_bytes)
    if not selected:
        return RepackResult(0, 0, 0, 0, (), 0)
    items = _live_audio_files(session, selected)
    source_packs = {pack.id: resolved_store.download(pack.path) for pack in selected}
    source_audio = {item.id: _slice(source_packs[item.bucket_file_id], item) for item in items}
    writer = AudioPackWriter(session, resolved_store, config)
    writes = {item.id: writer.append(source_audio[item.id]) for item in items}
    writer.flush()
    for item in items:
        write = writes[item.id]
        item.bucket_file_id = write.bucket_file.id
        item.bucket_file = write.bucket_file
        item.byte_offset = write.byte_offset
        item.byte_length = write.byte_length
    session.flush()
    created = {write.bucket_file.id: write.bucket_file for write in writes.values()}
    replacement_packs = {
        pack_id: resolved_store.download(pack.path)
        for pack_id, pack in created.items()
    }
    for item in items:
        write = writes[item.id]
        replacement = replacement_packs[write.bucket_file.id]
        start = write.byte_offset
        end = start + write.byte_length
        assert replacement[start:end] == source_audio[item.id], f"repack byte mismatch: {item.id}"
    deleted_paths = tuple(pack.path for pack in selected)
    for pack in selected:
        session.delete(pack)
    session.commit()
    for path in deleted_paths:
        resolved_store.delete(path)
    remaining = len(_legacy_packs(session, config.path_prefix, lock=False))
    return RepackResult(
        moved_audio_files=len(items),
        replaced_packs=len(selected),
        created_packs=len(created),
        bytes_verified=sum(len(data) for data in source_audio.values()),
        deleted_paths=deleted_paths,
        remaining_packs=remaining,
    )


def _legacy_packs(session: Session, prefix: str, lock: bool = True) -> list[BucketFile]:
    statement = select(BucketFile).where(BucketFile.path.like(f"{prefix}/%"))
    if lock:
        statement = statement.with_for_update()
    packs = session.execute(statement).scalars().all()
    return sorted(
        (pack for pack in packs if _is_legacy_path(pack.path, prefix)),
        key=lambda pack: pack.path,
    )


def _is_legacy_path(path: str, prefix: str) -> bool:
    relative = path.removeprefix(f"{prefix}/")
    return "/" not in relative and relative.endswith(".bin")


def _bounded_packs(packs: list[BucketFile], limit: int) -> list[BucketFile]:
    selected = []
    size = 0
    for pack in packs:
        if selected and size + pack.size > limit:
            break
        selected.append(pack)
        size += pack.size
    return selected


def _live_audio_files(session: Session, packs: list[BucketFile]) -> list[AudioFile]:
    pack_ids = [pack.id for pack in packs]
    statement = (
        select(AudioFile)
        .where(AudioFile.bucket_file_id.in_(pack_ids))
        .order_by(AudioFile.bucket_file_id, AudioFile.byte_offset)
        .with_for_update(of=AudioFile)
    )
    return list(session.execute(statement).scalars())


def _slice(pack_data: bytes, item: AudioFile) -> bytes:
    start = item.byte_offset
    return pack_data[start:start + item.byte_length]
