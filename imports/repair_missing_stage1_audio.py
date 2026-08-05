"""Repair backend audio packs whose metadata committed without their object bytes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import uuid
from collections import defaultdict
from pathlib import Path

from botocore.exceptions import ClientError

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioPartRead, AudioUpdate
from shared.db.datasets import crud as dataset_crud
from stage1_backend import ImportBatch, _audio_payload


STAGE_ROOT = Path(__file__).resolve().parent / "stage1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--dataset-name")
    return parser.parse_args()


def missing_pack_ids(
    dataset_name: str,
    slug: str,
    source_ids: set[str],
) -> set[uuid.UUID]:
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, dataset_name)
        assert dataset is not None, f"backend dataset not found: {dataset_name}"
        by_pack = defaultdict(list)
        items = dataset_crud.list_dataset_audio_files_by_stage1_slug(
            session,
            dataset.id,
            slug,
        )
        for item in items:
            if item.metadata_["stage1_source_id"] not in source_ids:
                continue
            by_pack[item.bucket_file_id].append(item)
        probes = []
        for pack_id, items in by_pack.items():
            last = max(
                items,
                key=lambda item: item.byte_offset + item.byte_length,
            )
            probes.append((pack_id, last.id, last.byte_length))

    def probe(
        item: tuple[uuid.UUID, uuid.UUID, int],
    ) -> uuid.UUID | None:
        pack_id, audio_id, byte_length = item
        with database_session() as session:
            try:
                audio_crud.read_audio_part(
                    session,
                    audio_id,
                    payload=AudioPartRead(start=byte_length - 1, length=1),
                )
            except ClientError as error:
                assert error.response["Error"]["Code"] in {
                    "InvalidRange",
                    "NoSuchKey",
                }
                return pack_id
            except AssertionError:
                return pack_id
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        return {pack_id for pack_id in executor.map(probe, probes) if pack_id is not None}


def repair(slug: str, dataset_name_override: str | None = None) -> None:
    root = STAGE_ROOT / slug
    manifest = json.loads((root / "data.json").read_text())
    dataset_name = (
        dataset_name_override
        if dataset_name_override is not None
        else manifest["dataset"]["name"]
    )
    records = {record["source_id"]: record for record in manifest["audio_files"]}
    journal = root / ".backend-verified-source-ids"
    verified = (
        set(journal.read_text(encoding="utf-8").splitlines())
        if journal.exists()
        else set()
    )
    missing = missing_pack_ids(
        dataset_name,
        slug,
        set(records).difference(verified),
    )
    print(f"MISSING_PACKS count={len(missing)}", flush=True)
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, dataset_name)
        assert dataset is not None
        scoped_items = dataset_crud.list_dataset_audio_files_by_stage1_slug(
            session,
            dataset.id,
            slug,
        )
        affected = [
            item
            for item in scoped_items
            if item.bucket_file_id in missing
        ]
        by_pack = defaultdict(list)
        for item in affected:
            by_pack[item.bucket_file_id].append(item)
    for pack_id, items in by_pack.items():
        payloads = {}
        for item in items:
            source_id = item.metadata_["stage1_source_id"]
            record = records[source_id]
            batch = ImportBatch(dataset.id, slug, root, (record,))
            create = _audio_payload(batch, record)
            payloads[item.id] = AudioUpdate(**create.model_dump(exclude={"waveform"}))
        with database_session() as session:
            audio_crud.bulk_update_audio_files(session, payloads)
        print(f"REPAIRED pack={pack_id} records={len(items)}", flush=True)


def main() -> None:
    values = arguments()
    repair(values.slug, values.dataset_name)


if __name__ == "__main__":
    main()
