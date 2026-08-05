import shutil
import time
from pathlib import Path

from imports.stage1.common.hf22_catalog import (
    LocaleSpec,
    load_specs,
    specs_for_part,
)
from imports.stage1.common.hf22_download import (
    DiskReserveReached,
    HuggingFaceClient,
    LocaleDeadlineExceeded,
    load_hf_token,
    shard_order,
)
from imports.stage1.common.hf22_prepare import (
    SpeakerBudget,
    process_shard,
    read_metadata,
)
from imports.stage1.common.hf22_state import ManifestStore, State, StatusStore


LOCALE_SECONDS = 45 * 60
CONVERSION_WORKERS = 8
MINIMUM_ACCEPTED_FRACTION = 0.25


def run_part(part: int, lane: int = 0, lanes: int = 1) -> None:
    assert part in (1, 2, 3)
    assert lanes > 0
    assert 0 <= lane < lanes
    repository_root = Path(__file__).resolve().parents[3]
    part_root = repository_root / "imports" / "stage1" / f"common_voice_part{part}"
    temporary_root = part_root / "tmp"
    catalog_root = temporary_root / f"hf22-catalog-lane{lane}"
    client = HuggingFaceClient(load_hf_token(repository_root), repository_root)
    catalog_deadline = time.monotonic() + 5 * 60
    stats_path, shards_path = client.catalog_files(
        catalog_root,
        catalog_deadline,
    )
    part_specs = specs_for_part(
        load_specs(repository_root, stats_path, shards_path),
        part,
    )
    specs = part_specs[lane::lanes]
    manifest = ManifestStore(part_root / "data.json")
    status = StatusStore(part_root, part)
    terminal_languages = status.normalize_exhausted(
        part_specs,
        MINIMUM_ACCEPTED_FRACTION,
    )
    try:
        for spec in specs:
            if spec.language in terminal_languages:
                continue
            _run_locale(
                spec,
                part_root,
                client,
                manifest,
                status,
            )
        total = manifest.validate()
        print(
            f"LANE_COMPLETE part={part} lane={lane}/{lanes} "
            f"manifest_records={total}",
            flush=True,
        )
    finally:
        shutil.rmtree(catalog_root, ignore_errors=True)


def _run_locale(
    spec: LocaleSpec,
    part_root: Path,
    client: HuggingFaceClient,
    manifest: ManifestStore,
    status: StatusStore,
) -> None:
    started = time.monotonic()
    deadline = started + LOCALE_SECONDS
    locale_tmp = part_root / "tmp" / f"hf22-{spec.hf_locale}"
    existing = manifest.locale_records(spec)
    duration = sum(record.duration for record in existing)
    failures = []
    shards_processed = 0
    metadata_counts: dict[str, int] = {}
    if duration >= spec.target_seconds:
        status.update(
            spec,
            State.COMPLETE_LOCAL,
            actual_hours=duration / 3600.0,
            records=len(existing),
            elapsed_seconds=0.0,
            shards_processed=0,
            metadata_rows={},
            conversion_failures=0,
        )
        return
    status.update(
        spec,
        State.ACTIVE,
        actual_hours=duration / 3600.0,
        records=len(existing),
        elapsed_seconds=0.0,
        shards_processed=0,
        metadata_rows={},
        conversion_failures=0,
    )
    state = State.FAILED
    error: str | None = None
    try:
        paths = client.metadata_files(spec, locale_tmp, deadline)
        metadata = read_metadata(spec, paths)
        metadata_counts = metadata.row_counts
        budget = SpeakerBudget(spec, metadata.speaker_count, existing)
        source_ids = {record.source_id for record in existing}
        records = list(existing)
        for task in shard_order(spec):
            if duration >= spec.target_seconds:
                break
            shard_path = client.shard(spec, task, locale_tmp, deadline)
            try:
                result = process_shard(
                    spec=spec,
                    shard_path=shard_path,
                    split=task.split,
                    metadata=metadata,
                    existing_source_ids=source_ids,
                    existing_duration=duration,
                    budget=budget,
                    wav_root=part_root / "wavs",
                    deadline=deadline,
                    workers=CONVERSION_WORKERS,
                )
            finally:
                shard_path.unlink(missing_ok=True)
            records.extend(result.records)
            failures.extend(result.failures)
            duration = sum(record.duration for record in records)
            shards_processed += 1
            manifest.merge(spec, records)
            status.update(
                spec,
                State.ACTIVE,
                actual_hours=duration / 3600.0,
                records=len(records),
                elapsed_seconds=time.monotonic() - started,
                shards_processed=shards_processed,
                metadata_rows=metadata_counts,
                conversion_failures=len(failures),
            )
            print(
                f"LOCALE_PROGRESS part={spec.part} language={spec.language} "
                f"hours={duration / 3600.0:.4f}/{spec.target_hours:.4f} "
                f"records={len(records)} shards={shards_processed}",
                flush=True,
            )
            if result.timed_out:
                raise LocaleDeadlineExceeded(spec.language)
        coverage = duration / spec.target_seconds
        state = (
            State.COMPLETE_LOCAL
            if coverage >= MINIMUM_ACCEPTED_FRACTION
            else State.NOT_POSSIBLE
        )
        if state == State.NOT_POSSIBLE:
            error = (
                "trusted train/dev/test shards exhausted at "
                f"{coverage:.1%} of target, below the 25% acceptance threshold"
            )
    except LocaleDeadlineExceeded as caught:
        state = State.TIME_LIMIT
        error = str(caught)
    except DiskReserveReached as caught:
        state = State.DISK_LIMIT
        error = str(caught)
    except Exception as caught:
        state = State.FAILED
        error = repr(caught)
    finally:
        shutil.rmtree(locale_tmp, ignore_errors=True)
        current = manifest.locale_records(spec)
        current_duration = sum(record.duration for record in current)
        status.update(
            spec,
            state,
            actual_hours=current_duration / 3600.0,
            records=len(current),
            elapsed_seconds=time.monotonic() - started,
            shards_processed=shards_processed,
            metadata_rows=metadata_counts,
            conversion_failures=len(failures),
            error=error,
            failure_samples=failures[:20],
        )
        print(
            f"LOCALE_DONE part={spec.part} language={spec.language} "
            f"state={state.value} hours={current_duration / 3600.0:.4f} "
            f"records={len(current)} elapsed={time.monotonic() - started:.1f}",
            flush=True,
        )
