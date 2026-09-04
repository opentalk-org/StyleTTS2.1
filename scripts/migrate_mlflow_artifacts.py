from base64 import b64encode
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import boto3
from clickhouse_connect.driver.client import Client
from tqdm import tqdm


METRICS_DIR_ENV = "METRICS_DIR"
S3_BUCKET_ENV = "RUNFLOW_S3_BUCKET"
ARTIFACT_PREFIX_ENV = "MLFLOW_ARTIFACT_PREFIX"
DEFAULT_ARTIFACT_PREFIX = "mlflow/"
STEP_PATTERN = re.compile(r"(?:^|[/_-])step[_-]?(\d+)(?:$|[/_.-])", re.IGNORECASE)


@dataclass(frozen=True)
class ArtifactSource:
    run_id: UUID
    path: str
    key: str
    size: int
    timestamp: datetime


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe MLflow artifact path: {value!r}")
    return path


def _step(path: str) -> int:
    matches = STEP_PATTERN.findall(path)
    return int(matches[-1]) if matches else 0


def _source(key: str, prefix: str, size: int, timestamp: datetime) -> ArtifactSource:
    experiment, run_hex, directory, path = key.removeprefix(prefix).split("/", 3)
    del experiment
    if directory != "artifacts":
        raise ValueError(f"unexpected MLflow object key: {key}")
    return ArtifactSource(UUID(hex=run_hex), path, key, size, timestamp)


def _sources(s3: Any, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            yield _source(key, prefix, item["Size"], item["LastModified"])


def _copy_artifact(
    s3: Any, bucket: str, source: ArtifactSource, metrics_dir: Path
) -> list[object]:
    relative = _safe_path(source.path)
    target = metrics_dir / str(source.run_id) / "artifacts" / Path(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.part")
    s3.download_file(bucket, source.key, str(staging))
    os.replace(staging, target)
    size = target.stat().st_size
    assert size == source.size, f"artifact size changed: {source.path}"
    content_type = mimetypes.guess_type(relative.name)[0] or "application/octet-stream"
    stored_value = (
        target.read_text(encoding="utf-8")
        if content_type == "application/json" or relative.name.endswith(".plot")
        else f"artifacts/{relative.as_posix()}"
    )
    return [
        uuid5(NAMESPACE_URL, f"mlflow-artifact:{source.run_id}:{relative.as_posix()}"),
        source.run_id,
        _step(relative.as_posix()),
        source.timestamp,
        relative.as_posix(),
        stored_value,
        content_type,
        size,
    ]


def migrate_artifacts(destination: Client, batch_size: int) -> int:
    metrics_dir = Path(os.environ[METRICS_DIR_ENV]).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    bucket = os.environ[S3_BUCKET_ENV]
    prefix = os.environ.get(ARTIFACT_PREFIX_ENV, DEFAULT_ARTIFACT_PREFIX)
    s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
    run_ids = {row[0] for row in destination.query("SELECT id FROM runs").result_rows}
    destination.command("TRUNCATE TABLE artifacts")
    columns = [
        "id",
        "run_id",
        "step",
        "timestamp",
        "name",
        "path",
        "content_type",
        "size_bytes",
    ]
    pending: list[list[object]] = []
    copied = 0
    with tqdm(desc="artifacts", unit="files") as progress:
        for source in _sources(s3, bucket, prefix):
            if source.run_id not in run_ids:
                continue
            pending.append(_copy_artifact(s3, bucket, source, metrics_dir))
            copied += 1
            progress.update()
            if len(pending) >= batch_size:
                destination.insert("artifacts", pending, column_names=columns)
                pending.clear()
        if pending:
            destination.insert("artifacts", pending, column_names=columns)
    count = destination.query("SELECT count() FROM artifacts").first_row[0]
    assert count == copied
    print(f"validated artifacts: {count}", flush=True)
    return count


def reconcile_checkpoint_runs(destination: Client) -> int:
    run_names: dict[str, list[UUID]] = {}
    for run_id, name in destination.query("SELECT id, name FROM runs").result_rows:
        run_names.setdefault(name, []).append(run_id)
    rows = destination.query(
        """
        SELECT id, updated_at, kind, name, step, path, size, content_hash, type,
               metadata, run_id, ancestor_asset_id
        FROM assets FINAL
        WHERE kind = 'checkpoint'
        """
    ).result_rows
    updates = []
    for row in rows:
        metadata = json.loads(row[9])
        state = metadata.get("state") or {}
        run_name = metadata.get("finetune_run_name") or state.get("finetune_run_name")
        matches = run_names.get(run_name, [])
        if len(matches) != 1:
            continue
        values = list(row)
        values[1] += timedelta(microseconds=1)
        values[10] = matches[0]
        updates.append(values)
    if updates:
        destination.insert(
            "assets",
            updates,
            column_names=[
                "id",
                "updated_at",
                "kind",
                "name",
                "step",
                "path",
                "size",
                "content_hash",
                "type",
                "metadata",
                "run_id",
                "ancestor_asset_id",
            ],
        )
    print(f"reconciled checkpoint runs: {len(updates)}", flush=True)
    return len(updates)


def backfill_inline_artifacts(destination: Client) -> int:
    metrics_dir = Path(os.environ[METRICS_DIR_ENV]).resolve()
    rows = destination.query(
        """
        SELECT id, run_id, path
        FROM artifacts
        WHERE content_type = 'application/json' OR endsWith(name, '.plot')
        """
    ).result_rows
    branches = []
    ids = []
    for artifact_id, run_id, path in rows:
        content = (metrics_dir / str(run_id) / path).read_bytes()
        encoded = b64encode(content).decode("ascii")
        ids.append(f"toUUID('{artifact_id}')")
        branches.append(f"id = toUUID('{artifact_id}'), base64Decode('{encoded}')")
    if not rows:
        return 0
    destination.command(
        f"ALTER TABLE artifacts UPDATE path = multiIf({', '.join(branches)}, path) "
        f"WHERE id IN ({', '.join(ids)})",
        settings={"max_query_size": 100_000_000, "mutations_sync": 2},
    )
    print(f"backfilled inline artifacts: {len(rows)}", flush=True)
    return len(rows)
