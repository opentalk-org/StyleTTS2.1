from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from runner.nodes.hetzner.ds_v2_rows import (
    METADATA_DIRECTORY,
    cached_remote_file,
    parse_metadata_row,
    validate_metadata_headers,
)


PARQUET_DIRECTORY = PurePosixPath("/home/ds_v2")
MAX_CACHED_METADATA_FILES = 4


@dataclass(frozen=True)
class DsV2MetadataRow:
    index: int
    remote_metadata_path: str
    remote_parquet_path: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class DsV2MetadataRows:
    remote_paths: list[str]
    host: str
    cache_dir: Path
    retries: int
    row_offset: int
    row_limit: int | None

    def __iter__(self) -> Iterator[DsV2MetadataRow]:
        skipped = 0
        emitted = 0
        try:
            for remote_path in self.remote_paths:
                if self.row_limit is not None and emitted >= self.row_limit:
                    return
                local_path = cached_remote_file(self.host, remote_path, self.cache_dir, self.retries)
                with local_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
                    reader = csv.DictReader(metadata_file)
                    validate_metadata_headers(reader.fieldnames, local_path)
                    for row_index, raw in enumerate(reader):
                        if skipped < self.row_offset:
                            skipped += 1
                            continue
                        if self.row_limit is not None and emitted >= self.row_limit:
                            return
                        row = parse_metadata_row(raw, local_path, row_index)
                        emitted += 1
                        yield DsV2MetadataRow(
                            row_index,
                            remote_path,
                            parquet_path_from_metadata(remote_path),
                            row,
                        )
                self.prune_cache()
        finally:
            self.prune_cache()

    def prune_cache(self) -> None:
        files = sorted(
            self.cache_dir.glob("ds_v2_*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[MAX_CACHED_METADATA_FILES:]:
            path.unlink()


def load_metadata_rows(
    host: str,
    cache_dir: Path,
    retries: int,
    row_offset: int,
    row_limit: int | None,
) -> DsV2MetadataRows:
    remote_paths = _list_metadata_files(host, retries)
    return DsV2MetadataRows(remote_paths, host, cache_dir, retries, row_offset, row_limit)


def _list_metadata_files(host: str, retries: int) -> list[str]:
    command = f"ls -1 {METADATA_DIRECTORY}/*.csv\n"
    last_detail = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "sftp", "-q", "-oBatchMode=yes", "-oConnectTimeout=30",
                "-oConnectionAttempts=3", "-oServerAliveInterval=15", "-b", "-", host,
            ],
            input=command,
            text=True,
            capture_output=True,
            check=False,
        )
        names = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().endswith(".csv") and not line.strip().startswith("sftp>")
        ]
        paths = sorted(
            str(PurePosixPath(name) if name.startswith("/") else METADATA_DIRECTORY / name)
            for name in names
        )
        if result.returncode == 0 and paths:
            return paths
        last_detail = f"exit={result.returncode} | {result.stderr.strip()} | {result.stdout.strip()}"
        if attempt < retries:
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(
        f"SFTP metadata listing failed after {retries} attempt(s) for {host}:{METADATA_DIRECTORY}: {last_detail}"
    )


def parquet_path_from_metadata(remote_metadata_path: str) -> str:
    metadata_name = PurePosixPath(remote_metadata_path).name
    suffix = "_metadata.csv"
    if not metadata_name.endswith(suffix):
        raise ValueError(f"ds_v2 metadata path must end in {suffix}: {remote_metadata_path}")
    parquet_name = f"{metadata_name[:-len(suffix)]}.parquet"
    return str(PARQUET_DIRECTORY / parquet_name)
