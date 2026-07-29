import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.datasets import crud as dataset_crud
from stage1_backend import (
    ImportBatch,
    _audio_payload,
    _batches,
    _stage_paths,
)
from stage1_backend_verify import _verify_row


@dataclass(frozen=True)
class VerifiedBatch:
    records: tuple[dict[str, object], ...]
    bytes_uploaded: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload, byte-verify, and prune staged audio batches"
    )
    parser.add_argument("slug")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _record_and_prune(root: Path, records: tuple[dict[str, object], ...]) -> None:
    journal = root / ".backend-verified-source-ids"
    with journal.open("a", encoding="utf-8") as output:
        for record in records:
            output.write(str(record["source_id"]) + "\n")
        output.flush()
        os.fsync(output.fileno())
    for record in records:
        (root / str(record["path"])).unlink(missing_ok=True)


def _upload_verified(batch: ImportBatch) -> VerifiedBatch:
    records = tuple(batch.records)
    payloads = [_audio_payload(batch, record) for record in records]
    with database_session() as session:
        items = audio_crud.bulk_create_audio_files(session, payloads)
        dataset_crud.bulk_add_audio_files_to_dataset(
            session,
            batch.dataset_id,
            [item.id for item in items],
        )
        stored = audio_crud.bulk_read_audio_files(
            session,
            [item.id for item in items],
        )
        for record, payload, item in zip(records, payloads, items, strict=True):
            _verify_row(batch.slug, batch.root, record, item, require_audio=True)
            assert stored[item.id] == payload.wav_bytes, (
                f"{record['source_id']}: uploaded audio bytes differ"
            )
    return VerifiedBatch(
        records=records,
        bytes_uploaded=sum(len(payload.wav_bytes) for payload in payloads),
    )


def _verify_existing(
    slug: str,
    root: Path,
    expected: dict[str, dict[str, object]],
    items: list[object],
    verified: set[str],
    batch_size: int,
    workers: int,
) -> None:
    pending = sorted(
        (
            item
            for item in items
            if item.metadata_["stage1_source_id"] not in verified
        ),
        key=lambda item: (item.bucket_file_id, item.byte_offset),
    )
    batches = [
        pending[start:start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _verify_existing_batch,
                slug,
                root,
                expected,
                batch,
            )
            for batch in batches
        ]
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{slug} existing",
            unit="batch",
        )
        for future in progress:
            _record_and_prune(root, future.result())
    for source_id in verified:
        (root / str(expected[source_id]["path"])).unlink(missing_ok=True)


def _verify_existing_batch(
    slug: str,
    root: Path,
    expected: dict[str, dict[str, object]],
    items: list[object],
) -> tuple[dict[str, object], ...]:
    with database_session() as session:
        stored = audio_crud.bulk_read_audio_files(
            session,
            [item.id for item in items],
        )
    records = []
    for item in items:
        source_id = item.metadata_["stage1_source_id"]
        record = expected[source_id]
        _verify_row(slug, root, record, item, require_audio=True)
        assert stored[item.id] == (root / str(record["path"])).read_bytes()
        records.append(record)
    return tuple(records)


def import_pruned(
    slug: str,
    dataset_name: str,
    workers: int,
    batch_size: int,
) -> None:
    path = _stage_paths([slug])[0]
    root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["audio_files"]
    expected = {record["source_id"]: record for record in records}
    journal = root / ".backend-verified-source-ids"
    verified = (
        set(journal.read_text(encoding="utf-8").splitlines())
        if journal.exists()
        else set()
    )
    assert verified <= set(expected)
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, dataset_name)
        assert dataset is not None, f"backend dataset not found: {dataset_name}"
        items = list(
            dataset_crud.list_dataset_audio_files_by_stage1_slug(
                session,
                dataset.id,
                slug,
            )
        )
    existing = {item.metadata_["stage1_source_id"] for item in items}
    assert existing <= set(expected)
    missing = [record for record in records if record["source_id"] not in existing]
    batches = _batches(dataset.id, path, missing, batch_size)
    uploaded = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_upload_verified, batch) for batch in batches]
        _verify_existing(
            slug,
            root,
            expected,
            items,
            verified,
            batch_size,
            workers,
        )
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=slug,
            unit="batch",
        )
        for future in progress:
            result = future.result()
            uploaded += result.bytes_uploaded
            _record_and_prune(root, result.records)
    elapsed = time.perf_counter() - started
    print(
        f"IMPORTED_VERIFIED_PRUNED {slug} records={len(missing)} "
        f"bytes={uploaded} elapsed={elapsed:.3f}",
        flush=True,
    )


def main() -> None:
    values = arguments()
    assert values.workers > 0
    assert values.batch_size > 0
    import_pruned(
        values.slug,
        values.dataset_name,
        values.workers,
        values.batch_size,
    )


if __name__ == "__main__":
    main()
