import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, UUID, uuid5

from clickhouse_connect.driver.client import Client
from mlflow import MlflowClient
from mlflow.entities import FileInfo
from tqdm import tqdm


MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
METRICS_DIR_ENV = "METRICS_DIR"
STEP_PATTERN = re.compile(r"(?:^|[/_-])step[_-]?(\d+)(?:$|[/_.-])", re.IGNORECASE)


@dataclass(frozen=True)
class ArtifactSource:
    run_id: UUID
    started_at_ms: int
    info: FileInfo


def _files(client: MlflowClient, run_id: str, path: str = "") -> list[FileInfo]:
    files: list[FileInfo] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.extend(_files(client, run_id, item.path))
        else:
            files.append(item)
    return files


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe MLflow artifact path: {value!r}")
    return path


def _step(path: str) -> int:
    matches = STEP_PATTERN.findall(path)
    return int(matches[-1]) if matches else 0


def _content_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _copy_artifact(
    client: MlflowClient, source: ArtifactSource, metrics_dir: Path
) -> list[object]:
    relative = _safe_path(source.info.path)
    run_root = metrics_dir / str(source.run_id)
    target = run_root / "artifacts" / Path(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlflow-artifact-") as temporary:
        downloaded = Path(
            client.download_artifacts(
                str(source.run_id), source.info.path, dst_path=temporary
            )
        )
        staging = target.with_name(f".{target.name}.part")
        shutil.copyfile(downloaded, staging)
        os.replace(staging, target)
    size = target.stat().st_size
    if source.info.file_size is not None:
        assert size == source.info.file_size, (
            f"artifact size changed: {source.info.path}"
        )
    artifact_path = f"artifacts/{relative.as_posix()}"
    return [
        uuid5(NAMESPACE_URL, f"mlflow-artifact:{source.run_id}:{relative.as_posix()}"),
        source.run_id,
        _step(relative.as_posix()),
        datetime.fromtimestamp(source.started_at_ms / 1000, UTC),
        relative.as_posix(),
        artifact_path,
        _content_type(relative.name),
        size,
    ]


def migrate_artifacts(destination: Client, batch_size: int) -> int:
    tracking_uri = os.environ[MLFLOW_TRACKING_URI_ENV]
    metrics_dir = Path(os.environ[METRICS_DIR_ENV]).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = [
        UUID(row[0]) for row in destination.query("SELECT id FROM runs").result_rows
    ]
    sources = []
    for run_id in runs:
        run = client.get_run(str(run_id))
        sources.extend(
            ArtifactSource(run_id, run.info.start_time, artifact)
            for artifact in _files(client, str(run_id))
        )
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
    with tqdm(sources, desc="artifacts", unit="files") as progress:
        for source in progress:
            pending.append(_copy_artifact(client, source, metrics_dir))
            if len(pending) >= batch_size:
                destination.insert("artifacts", pending, column_names=columns)
                pending.clear()
        if pending:
            destination.insert("artifacts", pending, column_names=columns)
    count = destination.query("SELECT count() FROM artifacts").first_row[0]
    assert count == len(sources)
    print(f"validated artifacts: {count}", flush=True)
    return count
