import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
from uuid import UUID

from shared.db.audio.timestamp_cleanup import crud
from shared.db.audio.timestamp_cleanup.schemas import CleanupBatchResult
from shared.db.connection import database_session
from shared.db.datasets import crud as dataset_crud


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = ROOT / ".data" / "youtube-timestamp-cleanup.json"


def main() -> int:
    options = _parse_args()
    checkpoint = _read_checkpoint(options.checkpoint)
    print(json.dumps({
        "event": "cleanup_initializing",
        "checkpoint": str(options.checkpoint),
        "resumed_after": str(checkpoint) if checkpoint is not None else None,
    }), flush=True)
    dataset_id = _youtube_dataset_id()
    total = _candidate_count(dataset_id, checkpoint, options.limit)
    print(json.dumps({
        "event": "cleanup_started",
        "total_candidates": total,
        "workers": options.workers,
        "batch_size": options.batch_size,
        "resumed_after": str(checkpoint) if checkpoint is not None else None,
    }), flush=True)
    batches = _candidate_batches(
        dataset_id,
        checkpoint,
        options.batch_size,
        options.limit,
    )
    context = mp.get_context("spawn")
    processed = pruned = already_pruned = missing = 0
    audits = []
    started = time.monotonic()
    worker_args = ((batch, options.audit_output is not None) for batch in batches)
    with context.Pool(options.workers) as pool:
        for result in pool.imap(_process_batch, worker_args, chunksize=1):
            processed += result.examined
            pruned += result.pruned
            already_pruned += result.already_pruned
            missing += result.missing_parakeet
            audits.extend(result.audits)
            _write_checkpoint(options.checkpoint, result.last_audio_file_id)
            elapsed = time.monotonic() - started
            rows_per_second = processed / elapsed
            remaining = total - processed
            print(json.dumps({
                "event": "batch_completed",
                "examined": processed,
                "total_candidates": total,
                "progress_percent": round(100 * processed / total, 3) if total else 100.0,
                "pruned": pruned,
                "already_pruned": already_pruned,
                "missing_parakeet": missing,
                "rows_per_second": round(rows_per_second, 3),
                "eta_seconds": round(remaining / rows_per_second, 1) if rows_per_second else None,
                "last_audio_file_id": str(result.last_audio_file_id),
            }), flush=True)
    if options.audit_output is not None:
        options.audit_output.parent.mkdir(parents=True, exist_ok=True)
        options.audit_output.write_text(
            json.dumps([asdict(item) for item in audits], default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        "event": "cleanup_completed",
        "examined": processed,
        "pruned": pruned,
        "already_pruned": already_pruned,
        "missing_parakeet": missing,
        "checkpoint": str(options.checkpoint),
    }), flush=True)
    return 0


def _candidate_count(dataset_id: UUID, after_id: UUID | None, limit: int | None) -> int:
    if limit is not None:
        return limit
    with database_session() as session:
        return crud.count_dataset_rows_after(session, dataset_id, after_id)


def _process_batch(arguments: tuple[list[UUID], bool]) -> CleanupBatchResult:
    audio_file_ids, capture_audit = arguments
    with database_session() as session:
        return crud.prune_timestamp_batch(session, audio_file_ids, capture_audit)


def _candidate_batches(
    dataset_id: UUID,
    after_id: UUID | None,
    batch_size: int,
    limit: int | None,
) -> Iterator[list[UUID]]:
    remaining = limit
    cursor = after_id
    while remaining is None or remaining > 0:
        page_size = batch_size if remaining is None else min(batch_size, remaining)
        with database_session() as session:
            batch = crud.list_timestamp_candidate_ids(session, dataset_id, cursor, page_size)
        if not batch:
            return
        yield batch
        cursor = batch[-1]
        if remaining is not None:
            remaining -= len(batch)


def _youtube_dataset_id() -> UUID:
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, "youtube")
        if dataset is None:
            raise KeyError("youtube dataset does not exist")
        return dataset.id


def _read_checkpoint(path: Path) -> UUID | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return UUID(payload["last_audio_file_id"])


def _write_checkpoint(path: Path, audio_file_id: UUID) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"last_audio_file_id": str(audio_file_id)}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove duplicated audio-level YouTube timestamps after guarded comparison",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--audit-output", type=Path)
    options = parser.parse_args()
    if options.workers <= 0 or options.batch_size <= 0:
        parser.error("workers and batch-size must be positive")
    if options.limit is not None and options.limit <= 0:
        parser.error("limit must be positive")
    return options


if __name__ == "__main__":
    raise SystemExit(main())
