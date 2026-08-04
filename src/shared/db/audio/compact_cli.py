import argparse
import fcntl
import logging
import os
from pathlib import Path

from sqlalchemy import func, select

from shared.db.assets.models import BucketFile
from shared.db.audio.maintenance import compact_audio_pack_batch
from shared.db.audio.pack_store import AudioPackConfig
from shared.db.connection import database_session
from shared.db.settings import crud as settings_crud

LOGGER = logging.getLogger(__name__)
LOCK_PATH = Path("/workspace/audio-pack-compact.lock")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = AudioPackConfig(target_pack_bytes=args.target_mib * 1024 * 1024)
    total_source = 0
    total_live = 0
    total_packs = 0
    total_replacements = 0
    total_audio = 0

    with LOCK_PATH.open("w") as lock_file, database_session() as session:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another audio pack compactor is already running") from error
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        store = settings_crud.object_store(session)
        store.test_connection()
        LOGGER.info("connected to object store; starting verified compaction")

        while True:
            result = compact_audio_pack_batch(
                session,
                store,
                config,
                max_source_bytes=args.batch_mib * 1024 * 1024,
            )
            if result.source_packs == 0:
                break
            total_source += result.source_bytes
            total_live += result.live_bytes
            total_packs += result.source_packs
            total_replacements += result.replacement_packs
            total_audio += result.moved_audio_files
            LOGGER.info(
                "committed batch: source_packs=%d replacement_packs=%d "
                "audio_files=%d source_mib=%.1f live_mib=%.1f",
                result.source_packs,
                result.replacement_packs,
                result.moved_audio_files,
                result.source_bytes / 1024 / 1024,
                result.live_bytes / 1024 / 1024,
            )

        orphan_count = session.scalar(
            select(func.count())
            .select_from(BucketFile)
            .where(~BucketFile.audio_files.any())
        )
    LOGGER.info(
        "complete: source_packs=%d replacement_packs=%d audio_files=%d "
        "source_gib=%.2f live_gib=%.2f retained_orphans=%d",
        total_packs,
        total_replacements,
        total_audio,
        total_source / 1024 / 1024 / 1024,
        total_live / 1024 / 1024 / 1024,
        orphan_count,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely compact audio packs in resumable batches")
    parser.add_argument("--target-mib", type=int, default=128)
    parser.add_argument("--batch-mib", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
