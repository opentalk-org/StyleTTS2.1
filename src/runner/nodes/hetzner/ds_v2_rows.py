from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


METADATA_DIRECTORY = PurePosixPath("/home/ds_v2_metadata")
IDENTITY_COLUMNS = ("chunk_index", "sample_index", "sample_start", "speaker_id")
CSV_METADATA_COLUMNS = (
    "duration",
    "chunk_index",
    "chunk_start",
    "chunk_end",
    "speaker_start",
    "speaker_end",
    "sample_index",
    "sample_start",
    "sample_end",
    "text_parakeet",
    "text_timestamps",
    "text_whisper",
    "text_canary",
    "text_src",
    "mos_score",
    "audio_path",
    "parquet_filename",
    "filename",
    "src_type",
    "src",
    "metadata",
    "speaker_id",
)


@dataclass(frozen=True)
class DsV2RowIdentity:
    chunk_index: int
    sample_index: int
    sample_start: float
    speaker_id: str


@dataclass(frozen=True)
class DsV2Row:
    index: int
    audio: bytes
    metadata: dict[str, str]


def cached_remote_file(host: str, remote_path: str, cache_dir: Path, retries: int) -> Path:
    cache_key = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:16]
    path = _cached_remote_file(host, remote_path, cache_dir, retries, cache_key)
    path.touch()
    return path


def validate_metadata_headers(fieldnames: list[str] | None, metadata_path: Path) -> None:
    _validate_headers(fieldnames, metadata_path)


def parse_metadata_row(row: dict[str, str | None], metadata_path: Path, row_index: int) -> dict[str, str]:
    return _metadata_row(row, metadata_path, row_index)


def _cached_remote_file(host: str, remote_path: str, cache_dir: Path, retries: int, pair_key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    basename = PurePosixPath(remote_path).name
    target = cache_dir / f"ds_v2_{pair_key}_{basename}"
    if target.exists() and target.stat().st_size > 0:
        return target
    _download_sftp_file(host, remote_path, target, retries)
    return target


def _download_sftp_file(host: str, remote_path: str, target: Path, retries: int) -> None:
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    last_detail = ""
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "sftp", "-q", "-oBatchMode=yes", "-oConnectTimeout=30",
                "-oConnectionAttempts=3", "-oServerAliveInterval=15", "-b", "-", host,
            ],
            input=f"get {remote_path} {temporary}\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and temporary.exists() and temporary.stat().st_size > 0:
            temporary.replace(target)
            return
        last_detail = _sftp_error_detail(result, temporary)
        temporary.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(
        f"SFTP download failed after {retries} attempt(s) for {host}:{remote_path}: {last_detail}"
    )


def _sftp_error_detail(result: subprocess.CompletedProcess[str], temporary: Path) -> str:
    parts = [
        f"exit={result.returncode}",
        f"tmp_exists={temporary.exists()}",
        f"tmp_bytes={temporary.stat().st_size if temporary.exists() else 0}",
        result.stderr.strip(),
        result.stdout.strip(),
    ]
    return " | ".join(part for part in parts if part)


def _validate_headers(fieldnames: list[str] | None, metadata_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"ds_v2 metadata CSV has no header: {metadata_path}")
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise ValueError(f"ds_v2 metadata CSV has duplicate headers {duplicates}: {metadata_path}")
    missing = sorted(set(CSV_METADATA_COLUMNS) - set(fieldnames))
    if missing:
        raise ValueError(f"ds_v2 metadata CSV missing required columns {missing}: {metadata_path}")


def _metadata_row(row: dict[str, str | None], metadata_path: Path, row_index: int) -> dict[str, str]:
    missing_values = [column for column in CSV_METADATA_COLUMNS if row[column] is None]
    if missing_values:
        raise ValueError(
            f"ds_v2 metadata CSV row {row_index} has missing values for {missing_values}: {metadata_path}"
        )
    return {column: str(row[column]) for column in CSV_METADATA_COLUMNS}


def _identity(row: dict[str, Any]) -> DsV2RowIdentity:
    return DsV2RowIdentity(
        chunk_index=int(row["chunk_index"]),
        sample_index=int(row["sample_index"]),
        sample_start=float(row["sample_start"]),
        speaker_id=str(row["speaker_id"]),
    )


def validate_identity(
    parquet_row: dict[str, Any],
    metadata: dict[str, str],
    parquet_path: Path,
    metadata_path: Path,
    row_index: int,
) -> None:
    parquet_identity = _identity(parquet_row)
    metadata_identity = _identity(metadata)
    for field_name in IDENTITY_COLUMNS:
        parquet_value = getattr(parquet_identity, field_name)
        metadata_value = getattr(metadata_identity, field_name)
        if parquet_value != metadata_value:
            raise ValueError(
                f"ds_v2 identity mismatch at row {row_index} field {field_name}: "
                f"parquet={parquet_value!r}, csv={metadata_value!r}; "
                f"parquet_path={parquet_path}, csv_path={metadata_path}"
            )
