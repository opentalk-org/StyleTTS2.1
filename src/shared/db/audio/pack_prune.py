from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_store import AudioPackConfig, AudioPackWriter, ObjectStore


@dataclass(frozen=True)
class PruneResult:
    pruned_paths: list[str]
    moved_audio_files: int


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
    live_items = _live_audio_files(session, candidates)
    pack_data = {pack.id: store.download(pack.path) for pack in candidates}
    writer = AudioPackWriter(session, store, config)
    for item in live_items:
        wav_bytes = _slice_audio(pack_data, item)
        write = writer.append(wav_bytes)
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
    return PruneResult(pruned_paths=pruned_paths, moved_audio_files=len(live_items))


def _fragmented_packs(session: Session, config: AudioPackConfig) -> list[BucketFile]:
    packs = session.execute(select(BucketFile).with_for_update()).scalars().all()
    return [pack for pack in packs if _is_fragmented(pack, config)]


def _is_fragmented(pack: BucketFile, config: AudioPackConfig) -> bool:
    if pack.size == 0:
        return True
    return pack.used_bytes / pack.size < config.prune_used_ratio


def _live_audio_files(session: Session, packs: list[BucketFile]) -> list[AudioFile]:
    pack_ids = [pack.id for pack in packs]
    statement = select(AudioFile).where(AudioFile.bucket_file_id.in_(pack_ids)).with_for_update(of=AudioFile)
    return list(session.execute(statement).scalars())


def _slice_audio(pack_data: dict[UUID, bytes], item: AudioFile) -> bytes:
    start = item.byte_offset
    end = start + item.byte_length
    return pack_data[item.bucket_file.id][start:end]
