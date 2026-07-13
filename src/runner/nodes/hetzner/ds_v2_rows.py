from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import pyarrow.parquet as pq


METADATA_DIRECTORY = PurePosixPath("/home/ds_v2_metadata")
MAX_CACHED_PAIRS = 4
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


def metadata_remote_path(remote_parquet_path: str) -> str:
    parquet = PurePosixPath(remote_parquet_path)
    if parquet.suffix != ".parquet":
        raise ValueError(f"ds_v2 remote path must end in .parquet: {remote_parquet_path}")
    return str(METADATA_DIRECTORY / f"{parquet.stem}_metadata.csv")


def load_rows(
    host: str,
    remote_parquet_path: str,
    cache_dir: Path,
    retries: int,
    row_offset: int,
    row_limit: int,
) -> list[DsV2Row]:
    remote_metadata_path = metadata_remote_path(remote_parquet_path)
    pair_key = hashlib.sha1(remote_parquet_path.encode("utf-8")).hexdigest()[:16]
    metadata_path = _cached_remote_file(host, remote_metadata_path, cache_dir, retries, pair_key)
    parquet_path = _cached_remote_file(host, remote_parquet_path, cache_dir, retries, pair_key)
    parquet_path.touch()
    metadata_path.touch()
    _prune_cache(cache_dir, pair_key)
    return load_local_pair(parquet_path, metadata_path, row_offset, row_limit)


def load_local_pair(parquet_path: Path, metadata_path: Path, row_offset: int, row_limit: int) -> list[DsV2Row]:
    parquet = pq.ParquetFile(parquet_path)
    required_parquet = {"audio", *IDENTITY_COLUMNS}
    missing_parquet = sorted(required_parquet - set(parquet.schema.names))
    if missing_parquet:
        raise ValueError(f"Parquet missing required columns {missing_parquet}: {parquet_path}")

    selected_metadata: dict[int, dict[str, str]] = {}
    parquet_identities = _parquet_identities(parquet)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        _validate_headers(reader.fieldnames, metadata_path)
        csv_count = 0
        for row_index, csv_row in enumerate(reader):
            parquet_row = next(parquet_identities, None)
            if parquet_row is None:
                raise ValueError(
                    f"ds_v2 row count mismatch: parquet={parquet.metadata.num_rows}, csv has more rows; "
                    f"parquet={parquet_path}, csv={metadata_path}"
                )
            metadata = _metadata_row(csv_row, metadata_path, row_index)
            _validate_identity(parquet_row, metadata, parquet_path, metadata_path, row_index)
            if row_offset <= row_index < row_offset + row_limit:
                selected_metadata[row_index] = metadata
            csv_count += 1

    if csv_count != parquet.metadata.num_rows:
        raise ValueError(
            f"ds_v2 row count mismatch: parquet={parquet.metadata.num_rows}, csv={csv_count}; "
            f"parquet={parquet_path}, csv={metadata_path}"
        )
    audio_by_index = _selected_audio(parquet, row_offset, row_limit)
    return [
        DsV2Row(index=index, audio=audio_by_index[index], metadata=metadata)
        for index, metadata in selected_metadata.items()
    ]


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


def _prune_cache(cache_dir: Path, current_pair_key: str) -> None:
    parquets = sorted(cache_dir.glob("ds_v2_*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    others = [parquet for parquet in parquets if parquet.name.split("_", 3)[2] != current_pair_key]
    for parquet in others[MAX_CACHED_PAIRS - 1:]:
        pair_key = parquet.name.split("_", 3)[2]
        parquet.unlink()
        for metadata in cache_dir.glob(f"ds_v2_{pair_key}_*.csv"):
            metadata.unlink()


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


def _parquet_identities(parquet: pq.ParquetFile) -> Iterator[dict[str, Any]]:
    for batch in parquet.iter_batches(batch_size=1024, columns=list(IDENTITY_COLUMNS)):
        yield from batch.to_pylist()


def _identity(row: dict[str, Any]) -> DsV2RowIdentity:
    return DsV2RowIdentity(
        chunk_index=int(row["chunk_index"]),
        sample_index=int(row["sample_index"]),
        sample_start=float(row["sample_start"]),
        speaker_id=str(row["speaker_id"]),
    )


def _validate_identity(
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


def _selected_audio(parquet: pq.ParquetFile, row_offset: int, row_limit: int) -> dict[int, bytes]:
    selected: dict[int, bytes] = {}
    absolute_index = 0
    end = row_offset + row_limit
    for batch in parquet.iter_batches(batch_size=max(1, min(64, row_limit)), columns=["audio"]):
        for row in batch.to_pylist():
            if row_offset <= absolute_index < end:
                audio = row["audio"]
                if not isinstance(audio, (bytes, bytearray)):
                    raise ValueError(f"ds_v2 row {absolute_index} has no audio bytes")
                selected[absolute_index] = bytes(audio)
            absolute_index += 1
            if absolute_index >= end:
                return selected
    return selected
