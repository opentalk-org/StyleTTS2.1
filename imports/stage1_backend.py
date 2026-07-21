import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from shared.audio_annotations import AudioAnnotations
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioCreate
from shared.db.datasets import crud as dataset_crud
from shared.db.datasets.schemas import DatasetCreate


STAGE_ROOT = Path(__file__).resolve().parent / "stage1"
DEFAULT_WORKERS = 12
DEFAULT_BATCH_SIZE = 128
DELETE_BATCH_SIZE = 5_000


@dataclass(frozen=True)
class ImportBatch:
    dataset_id: uuid.UUID
    slug: str
    root: Path
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ImportResult:
    slug: str
    count: int
    bytes_uploaded: int


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk-import Stage 1 audio through shared CRUD facades")
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("slugs", nargs="*")
    import_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    import_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    import_parser.add_argument("--min-realtime", type=float, default=0.0)
    subparsers.add_parser("clear")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("slugs", nargs="*")
    return parser.parse_args()


def _stage_paths(slugs: list[str]) -> list[Path]:
    paths = sorted(STAGE_ROOT.glob("*/data.json"))
    if slugs:
        requested = set(slugs)
        paths = [path for path in paths if path.parent.name in requested]
        found = {path.parent.name for path in paths}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"staged datasets not found: {missing}")
    return paths


def _dataset(session, name: str):
    matches = [item for item in dataset_crud.list_datasets(session) if item.name == name]
    if len(matches) > 1:
        raise ValueError(f"multiple backend datasets named {name!r}")
    if matches:
        return matches[0]
    return dataset_crud.create_dataset(session, DatasetCreate(name=name))


def _batches(
    dataset_id: uuid.UUID,
    path: Path,
    records: list[dict[str, Any]],
    batch_size: int,
) -> list[ImportBatch]:
    return [
        ImportBatch(dataset_id, path.parent.name, path.parent, tuple(records[start:start + batch_size]))
        for start in range(0, len(records), batch_size)
    ]


def _alignment(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "word": str(item["word"] if "word" in item else item["text"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in items
    ]


def _segments(slug: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for index, source in enumerate(record["segments"]):
        metadata = {
            "source": source["source"],
            "stage1_alignment": source["alignment"],
        }
        annotations = AudioAnnotations(
            speaker_id=record["speaker_id"],
            score=source["score"],
            accuracy=source["accuracy"],
            metadata=metadata,
        )
        segment_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"runflow-stage1:{slug}:{record['source_id']}:{index}",
        )
        segments.append({
            "id": str(segment_id),
            "start": float(source["start"]),
            "end": float(source["end"]),
            "text": str(source["text"]),
            "phon": str(source["phon"] if "phon" in source else ""),
            "annotations": annotations.model_dump(mode="json"),
            "type_": "dataset",
            "alignment": _alignment(source["alignment"]),
        })
    return segments


def _audio_payload(batch: ImportBatch, record: dict[str, Any]) -> AudioCreate:
    wav_path = batch.root / str(record["path"])
    metadata = {
        **record["metadata"],
        "stage1_dataset": batch.slug,
        "stage1_source_id": record["source_id"],
        "stage1_path": record["path"],
    }
    return AudioCreate(
        name=f"{batch.slug}/{wav_path.name}",
        wav_bytes=wav_path.read_bytes(),
        duration=float(record["duration"]),
        annotations=AudioAnnotations(
            speaker_id=record["speaker_id"],
            score=record["score"],
            accuracy=record["accuracy"],
            metadata=metadata,
        ),
        language=record["language"],
        style_prompt=record["style_prompt"],
        voice_prompt=record["voice_prompt"],
        segments=_segments(batch.slug, record),
        virtual=False,
    )


def _import_batch(batch: ImportBatch) -> ImportResult:
    payloads = [_audio_payload(batch, record) for record in batch.records]
    byte_count = sum(len(payload.wav_bytes) for payload in payloads)
    with database_session() as session:
        items = audio_crud.bulk_create_audio_files(session, payloads)
        dataset_crud.bulk_add_audio_files_to_dataset(
            session, batch.dataset_id, [item.id for item in items],
        )
    return ImportResult(batch.slug, len(items), byte_count)


def import_stage(slugs: list[str], workers: int, batch_size: int, min_realtime: float) -> None:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch size must be positive")
    overall_started = time.perf_counter()
    overall_duration = 0.0
    for path in _stage_paths(slugs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["audio_files"]
        duration = sum(float(record["duration"]) for record in records)
        with database_session() as session:
            dataset = _dataset(session, str(payload["dataset"]["name"]))
        batches = _batches(dataset.id, path, records, batch_size)
        imported = 0
        uploaded = 0
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results: Iterator[ImportResult] = executor.map(_import_batch, batches)
            for result in tqdm(results, total=len(batches), desc=path.parent.name, unit="batch"):
                imported += result.count
                uploaded += result.bytes_uploaded
        assert imported == len(records), f"{path.parent.name}: imported {imported} of {len(records)}"
        elapsed = time.perf_counter() - started
        realtime = duration / elapsed
        overall_duration += duration
        print(
            f"IMPORTED {path.parent.name} records={imported} bytes={uploaded} "
            f"seconds={duration:.6f} elapsed={elapsed:.6f} realtime={realtime:.3f}x",
            flush=True,
        )
    overall_elapsed = time.perf_counter() - overall_started
    overall_realtime = overall_duration / overall_elapsed
    print(
        f"IMPORT_TOTAL seconds={overall_duration:.6f} elapsed={overall_elapsed:.6f} "
        f"realtime={overall_realtime:.3f}x",
        flush=True,
    )
    if overall_realtime < min_realtime:
        raise RuntimeError(
            f"import throughput {overall_realtime:.3f}x is below required {min_realtime:.3f}x"
        )


def clear_audio() -> None:
    with database_session() as session:
        ids = audio_crud.search_audio_file_ids(session, "", "all")
    for start in tqdm(range(0, len(ids), DELETE_BATCH_SIZE), desc="delete audio", unit="batch"):
        with database_session() as session:
            audio_crud.bulk_delete_audio_files(session, ids[start:start + DELETE_BATCH_SIZE])
    with database_session() as session:
        removed_packs = audio_crud.purge_orphaned_audio_packs(session)
    print(f"CLEARED audio={len(ids)} packs={len(removed_packs)}", flush=True)


def verify(slugs: list[str]) -> None:
    staged = {
        json.loads(path.read_text(encoding="utf-8"))["dataset"]["name"]: (
            path.parent.name,
            len(json.loads(path.read_text(encoding="utf-8"))["audio_files"]),
        )
        for path in _stage_paths(slugs)
    }
    with database_session() as session:
        counts = {dataset.name: count for dataset, count in dataset_crud.list_dataset_file_counts(session)}
    for name, (slug, expected) in staged.items():
        actual = counts[name]
        assert actual == expected, f"{slug}: backend={actual}, staged={expected}"
        print(f"VERIFIED {slug} records={actual}")


def main() -> None:
    arguments = _arguments()
    if arguments.command == "import":
        import_stage(
            arguments.slugs,
            arguments.workers,
            arguments.batch_size,
            arguments.min_realtime,
        )
    elif arguments.command == "clear":
        clear_audio()
    elif arguments.command == "verify":
        verify(arguments.slugs)
    else:
        raise ValueError(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    main()
