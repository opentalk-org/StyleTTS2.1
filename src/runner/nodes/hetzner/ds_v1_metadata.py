from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from runner.nodes.hetzner.ds_v2_rows import (
    METADATA_DIRECTORY,
    cached_remote_file,
    parse_metadata_row,
    validate_metadata_headers,
)


@dataclass(frozen=True)
class DsV2Sample:
    row_index: int
    values: dict[str, str]


@dataclass(frozen=True)
class DsV2MetadataIndex:
    remote_path: str
    samples_by_recording: dict[str, tuple[DsV2Sample, ...]]


def metadata_path_for_v1(remote_parquet_path: str) -> str:
    stem = PurePosixPath(remote_parquet_path).stem
    return str(METADATA_DIRECTORY / f"{stem}_processed_metadata.csv")


def load_metadata_index(
    host: str,
    remote_v1_path: str,
    cache_dir: Path,
    retries: int,
    check_cancel: Callable[[], None],
) -> DsV2MetadataIndex:
    remote_path = metadata_path_for_v1(remote_v1_path)
    local_path = cached_remote_file(host, remote_path, cache_dir, retries)
    return read_metadata_index(local_path, remote_path, check_cancel)


def read_metadata_index(
    path: Path,
    remote_path: str,
    check_cancel: Callable[[], None],
) -> DsV2MetadataIndex:
    grouped: defaultdict[str, list[DsV2Sample]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        validate_metadata_headers(reader.fieldnames, path)
        for row_index, raw in enumerate(reader):
            check_cancel()
            values = parse_metadata_row(raw, path, row_index)
            audio_path = _normalized_name(values["audio_path"])
            filename = _normalized_name(values["filename"])
            if audio_path != filename:
                raise ValueError(
                    f"ds_v2 metadata row {row_index} has conflicting recording names: "
                    f"audio_path={audio_path!r}, filename={filename!r}; path={remote_path}"
                )
            grouped[audio_path].append(DsV2Sample(row_index, values))
    return DsV2MetadataIndex(
        remote_path,
        {key: tuple(samples) for key, samples in grouped.items()},
    )


def matching_samples(
    index: DsV2MetadataIndex,
    row: dict[str, Any],
    row_index: int,
) -> tuple[DsV2Sample, ...]:
    opus_file = _optional_name(row, "opus_file")
    video_id = _optional_text(row, "video_id")
    video_file = f"{video_id}.opus" if video_id is not None else None
    if opus_file is not None and video_file is not None and opus_file != video_file:
        raise ValueError(
            f"ds_v1 row {row_index} has conflicting recording identities: "
            f"opus_file={opus_file!r}, video_id={video_id!r}"
        )
    recording_name = opus_file or video_file
    if recording_name is None:
        raise ValueError(f"ds_v1 row {row_index} has neither opus_file nor video_id")
    if recording_name in index.samples_by_recording:
        return index.samples_by_recording[recording_name]
    return ()


def merge_recording_metadata(
    metadata: dict[str, Any],
    samples: tuple[DsV2Sample, ...],
    remote_path: str,
) -> dict[str, Any]:
    merged = dict(metadata)
    merged["ds_v2_sample_count"] = len(samples)
    if not samples:
        return merged
    source_metadata = _source_metadata(samples[0])
    for sample in samples[1:]:
        candidate = _source_metadata(sample)
        if candidate != source_metadata:
            _raise_metadata_conflict(metadata, samples[0], source_metadata, sample, candidate)
    v1_video_id = _optional_text(metadata, "video_id")
    v2_video_id = _optional_text(source_metadata, "video_id")
    if v1_video_id is not None and v2_video_id is not None and v1_video_id != v2_video_id:
        source_row = metadata["source_row_index"] if "source_row_index" in metadata else "unknown"
        raise ValueError(
            f"ds_v1 row {source_row} video_id conflicts with ds_v2 row {samples[0].row_index}: "
            f"ds_v1={v1_video_id!r}, ds_v2={v2_video_id!r}"
        )
    for key, value in source_metadata.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    merged["ds_v2_metadata_path"] = remote_path
    return merged


def _raise_metadata_conflict(
    metadata: dict[str, Any],
    first: DsV2Sample,
    first_metadata: dict[str, Any],
    other: DsV2Sample,
    other_metadata: dict[str, Any],
) -> None:
    keys = sorted(
        key
        for key in first_metadata.keys() | other_metadata.keys()
        if (first_metadata[key] if key in first_metadata else None)
        != (other_metadata[key] if key in other_metadata else None)
    )
    source_row = metadata["source_row_index"] if "source_row_index" in metadata else "unknown"
    raise ValueError(
        f"ds_v1 row {source_row} has conflicting ds_v2 recording metadata at rows "
        f"{first.row_index} and {other.row_index} for keys {keys}"
    )


def _source_metadata(sample: DsV2Sample) -> dict[str, Any]:
    try:
        value = json.loads(sample.values["metadata"])
    except json.JSONDecodeError as error:
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has invalid metadata JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"ds_v2 metadata row {sample.row_index} metadata must be an object")
    return value


def _optional_name(row: dict[str, Any], key: str) -> str | None:
    value = _optional_text(row, key)
    return _normalized_name(value) if value is not None else None


def _optional_text(row: dict[str, Any], key: str) -> str | None:
    if key not in row or row[key] is None:
        return None
    text = str(row[key]).strip()
    return text or None


def _normalized_name(value: str) -> str:
    name = PurePosixPath(value).name
    if not name:
        raise ValueError(f"recording path has no filename: {value!r}")
    return name
