from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Protocol, Sequence

import pyarrow.parquet as pq

from runner.nodes.hetzner.ds_v2_rows import (
    IDENTITY_COLUMNS,
    DsV2Row,
    cached_remote_file,
    validate_identity,
)


MAX_CACHED_PARQUETS = 4


class SelectedMetadataRow(Protocol):
    index: int
    remote_metadata_path: str
    remote_parquet_path: str
    metadata: dict[str, str]


def load_selected_audio_rows(
    host: str,
    rows: Sequence[SelectedMetadataRow],
    cache_dir: Path,
    retries: int,
) -> list[DsV2Row]:
    local_paths = {
        remote_path: cached_remote_file(host, remote_path, cache_dir, retries)
        for remote_path in {row.remote_parquet_path for row in rows}
    }
    try:
        return load_selected_local_audio_rows(local_paths, rows)
    finally:
        _prune_parquet_cache(cache_dir)


def load_selected_local_audio_rows(
    local_paths: dict[str, Path],
    rows: Sequence[SelectedMetadataRow],
) -> list[DsV2Row]:
    grouped: dict[str, list[tuple[int, SelectedMetadataRow]]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[row.remote_parquet_path].append((position, row))

    loaded: dict[int, DsV2Row] = {}
    for remote_path, selections in grouped.items():
        _load_parquet_selections(local_paths[remote_path], selections, loaded)
    return [loaded[position] for position in range(len(rows))]


def _load_parquet_selections(
    parquet_path: Path,
    selections: list[tuple[int, SelectedMetadataRow]],
    loaded: dict[int, DsV2Row],
) -> None:
    parquet = pq.ParquetFile(parquet_path)
    required = {"audio", *IDENTITY_COLUMNS}
    missing = sorted(required - set(parquet.schema.names))
    if missing:
        raise ValueError(f"Parquet missing required columns {missing}: {parquet_path}")

    by_index: dict[int, list[tuple[int, SelectedMetadataRow]]] = defaultdict(list)
    for position, row in selections:
        by_index[row.index].append((position, row))
    maximum = max(by_index)
    absolute_index = 0
    columns = ["audio", *IDENTITY_COLUMNS]
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for parquet_row in batch.to_pylist():
            for position, selected in by_index.get(absolute_index, []):
                validate_identity(
                    parquet_row,
                    selected.metadata,
                    parquet_path,
                    Path(selected.remote_metadata_path),
                    absolute_index,
                )
                audio = parquet_row["audio"]
                if not isinstance(audio, (bytes, bytearray)):
                    raise ValueError(f"ds_v2 row {absolute_index} has no audio bytes: {parquet_path}")
                loaded[position] = DsV2Row(absolute_index, bytes(audio), selected.metadata)
            absolute_index += 1
            if absolute_index > maximum:
                break
        if absolute_index > maximum:
            break

    missing_indices = sorted(index for index in by_index if index >= absolute_index)
    if missing_indices:
        raise ValueError(f"ds_v2 row {missing_indices[0]} is out of range: {parquet_path}")


def _prune_parquet_cache(cache_dir: Path) -> None:
    paths = sorted(
        cache_dir.glob("ds_v2_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths[MAX_CACHED_PARQUETS:]:
        path.unlink()
