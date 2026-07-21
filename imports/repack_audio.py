import argparse
import time

from tqdm import tqdm

from shared.db import database_session
from shared.db.audio.repack_crud import repack_legacy_audio_packs


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack legacy audio objects into sharded packs")
    parser.add_argument("--max-source-mib", type=int, default=512)
    return parser.parse_args()


def repack(max_source_bytes: int) -> None:
    if max_source_bytes < 1:
        raise ValueError("repack source-byte limit must be positive")
    started = time.perf_counter()
    moved_audio = 0
    replaced_packs = 0
    created_packs = 0
    verified_bytes = 0
    with tqdm(desc="repack audio", unit="pack") as progress:
        while True:
            with database_session() as session:
                result = repack_legacy_audio_packs(
                    session,
                    max_source_bytes=max_source_bytes,
                )
            if result.replaced_packs == 0:
                break
            moved_audio += result.moved_audio_files
            replaced_packs += result.replaced_packs
            created_packs += result.created_packs
            verified_bytes += result.bytes_verified
            progress.update(result.replaced_packs)
            progress.set_postfix(
                audio=moved_audio,
                created=created_packs,
                remaining=result.remaining_packs,
            )
    elapsed = time.perf_counter() - started
    print(
        f"REPACKED audio={moved_audio} replaced_packs={replaced_packs} "
        f"created_packs={created_packs} verified_bytes={verified_bytes} "
        f"elapsed={elapsed:.3f}",
        flush=True,
    )


def main() -> None:
    arguments = _arguments()
    repack(arguments.max_source_mib * 1024 * 1024)


if __name__ == "__main__":
    main()
