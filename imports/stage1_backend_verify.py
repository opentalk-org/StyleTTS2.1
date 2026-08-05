import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.datasets import crud as dataset_crud


READ_PACK_BATCH_SIZE = 40


def _same_number(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)


def _expected_segment(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    alignment = [
        {
            "word": str(item["word"] if "word" in item else item["text"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in source["alignment"]
    ]
    return {
        "start": float(source["start"]),
        "end": float(source["end"]),
        "text": str(source["text"]),
        "phon": str(source["phon"] if "phon" in source else ""),
        "type_": "dataset",
        "alignment": alignment,
        "annotations": {
            "speaker_id": record["speaker_id"],
            "score": source["score"],
            "accuracy": source["accuracy"],
            "metadata": {
                "source": source["source"],
                "stage1_alignment": source["alignment"],
            },
        },
    }


def _verify_segments(record: dict[str, Any], actual: list[dict[str, Any]]) -> None:
    assert len(actual) == len(record["segments"])
    for source, stored in zip(record["segments"], actual, strict=True):
        expected = _expected_segment(record, source)
        comparable = {key: stored[key] for key in expected}
        assert comparable == expected, f"{record['source_id']}: segment mismatch"


def _verify_row(
    slug: str, root: Path, record: dict[str, Any], item: Any, require_audio: bool,
) -> None:
    source_id = record["source_id"]
    expected_metadata = {
        **record["metadata"],
        "stage1_dataset": slug,
        "stage1_source_id": source_id,
        "stage1_path": record["path"],
    }
    assert item.name == f"{slug}/{Path(record['path']).name}"
    assert _same_number(item.duration, record["duration"])
    assert item.speaker_id == record["speaker_id"]
    assert _same_number(item.score, record["score"])
    assert _same_number(item.accuracy, record["accuracy"])
    assert item.language == record["language"]
    assert item.style_prompt == record["style_prompt"]
    assert item.voice_prompt == record["voice_prompt"]
    assert item.metadata_ == expected_metadata
    assert item.virtual is False
    _verify_segments(record, item.segments)
    if require_audio:
        assert (root / record["path"]).is_file(), f"{source_id}: staged audio missing"


def _load_verified(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text(encoding="utf-8").splitlines())


def _record_verified(path: Path, source_ids: list[str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for source_id in source_ids:
            output.write(source_id + "\n")
        output.flush()
        os.fsync(output.fileno())


def verify_stage_paths(paths: list[Path], prune_verified: bool = False) -> None:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = path.parent.name
        records = payload["audio_files"]
        expected = {record["source_id"]: record for record in records}
        journal = path.parent / ".backend-verified-source-ids"
        verified = _load_verified(journal) if prune_verified else set()
        assert verified <= set(expected), f"{slug}: verification journal contains unknown IDs"
        if prune_verified:
            for source_id in verified:
                (path.parent / expected[source_id]["path"]).unlink(missing_ok=True)
        with database_session() as session:
            dataset = dataset_crud.get_dataset_by_name(session, payload["dataset"]["name"])
            assert dataset is not None, f"{slug}: backend dataset not found"
            items = dataset_crud.list_dataset_audio_files_by_stage1_slug(
                session,
                dataset.id,
                slug,
            )
            actual = {item.metadata_["stage1_source_id"]: item for item in items}
            assert set(actual) == set(expected), f"{slug}: source IDs differ"
            for source_id, item in actual.items():
                _verify_row(
                    slug, path.parent, expected[source_id], item,
                    require_audio=source_id not in verified,
                )
            ids = list(item.id for item in items)
            items_by_id = {item.id: item for item in items}
            ids_by_pack = defaultdict(list)
            for location in audio_crud.audio_bucket_locations(session, ids):
                ids_by_pack[location.bucket_file_id].append(location.audio_file_id)
            packs = list(ids_by_pack.values())
            expected_bytes = sum(int(item.byte_length) for item in items)
            verified_bytes = sum(
                int(actual[source_id].byte_length) for source_id in verified
            )
            for start in range(0, len(packs), READ_PACK_BATCH_SIZE):
                batch_ids = [audio_id for pack in packs[start:start + READ_PACK_BATCH_SIZE] for audio_id in pack]
                pending_ids = [
                    audio_id for audio_id in batch_ids
                    if items_by_id[audio_id].metadata_["stage1_source_id"] not in verified
                ]
                stored = audio_crud.bulk_read_audio_files(session, pending_ids)
                verified_records: list[dict[str, Any]] = []
                for audio_id in pending_ids:
                    item = items_by_id[audio_id]
                    record = expected[item.metadata_["stage1_source_id"]]
                    staged = (path.parent / record["path"]).read_bytes()
                    assert stored[audio_id] == staged, f"{record['source_id']}: audio bytes differ"
                    verified_bytes += len(staged)
                    verified_records.append(record)
                if prune_verified and verified_records:
                    source_ids = [record["source_id"] for record in verified_records]
                    _record_verified(journal, source_ids)
                    verified.update(source_ids)
                    for record in verified_records:
                        (path.parent / record["path"]).unlink()
                print(
                    f"VERIFY_PROGRESS {slug} bytes={verified_bytes}/{expected_bytes} "
                    f"packs={min(start + READ_PACK_BATCH_SIZE, len(packs))}/{len(packs)}",
                    flush=True,
                )
        print(f"VERIFIED {slug} records={len(records)} fields=all audio_bytes=all", flush=True)
