import argparse
import json
from pathlib import Path

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.datasets import crud as dataset_crud
from stage1_backend_verify import _verify_row


STAGE_ROOT = Path(__file__).resolve().parent / "stage1"
DELETE_BATCH_SIZE = 5_000


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete stale Stage 1 backend rows before a corrected re-import"
    )
    parser.add_argument("slug")
    return parser.parse_args()


def find_stale_rows(
    slug: str,
    root: Path,
    records: dict[str, dict[str, object]],
    items: list[object],
) -> tuple[list[object], set[str]]:
    stale_ids = []
    stale_sources = set()
    for item in items:
        source_id = item.metadata_["stage1_source_id"]
        record = records.get(source_id)
        try:
            assert record is not None, f"{source_id}: no longer selected"
            _verify_row(slug, root, record, item, require_audio=False)
        except AssertionError:
            stale_ids.append(item.id)
            stale_sources.add(source_id)
    return stale_ids, stale_sources


def update_verification_journal(root: Path, stale_sources: set[str]) -> None:
    journal = root / ".backend-verified-source-ids"
    if not journal.exists():
        return
    retained = [
        source_id
        for source_id in journal.read_text(encoding="utf-8").splitlines()
        if source_id not in stale_sources
    ]
    temporary = journal.with_suffix(".tmp")
    text = "".join(f"{source_id}\n" for source_id in retained)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(journal)


def reconcile(slug: str) -> None:
    root = STAGE_ROOT / slug
    payload = json.loads((root / "data.json").read_text(encoding="utf-8"))
    records = {record["source_id"]: record for record in payload["audio_files"]}
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, payload["dataset"]["name"])
        assert dataset is not None, f"{slug}: backend dataset not found"
        items = list(
            dataset_crud.list_dataset_audio_files_by_stage1_slug(
                session,
                dataset.id,
                slug,
            )
        )
        stale_ids, stale_sources = find_stale_rows(slug, root, records, items)
    print(
        f"RECONCILE_AUDIT {slug} backend={len(items)} "
        f"expected={len(records)} stale={len(stale_ids)}",
        flush=True,
    )
    for start in range(0, len(stale_ids), DELETE_BATCH_SIZE):
        with database_session() as session:
            audio_crud.bulk_delete_audio_files(
                session, stale_ids[start:start + DELETE_BATCH_SIZE]
            )
        print(
            f"RECONCILE_DELETE {slug} "
            f"deleted={min(start + DELETE_BATCH_SIZE, len(stale_ids))}/{len(stale_ids)}",
            flush=True,
        )
    update_verification_journal(root, stale_sources)
    with database_session() as session:
        removed_packs = audio_crud.purge_orphaned_audio_packs(session)
    print(
        f"RECONCILED {slug} deleted={len(stale_ids)} "
        f"orphaned_packs={len(removed_packs)}",
        flush=True,
    )


def main() -> None:
    reconcile(arguments().slug)


if __name__ == "__main__":
    main()
