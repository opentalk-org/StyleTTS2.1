from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq

from runner.nodes.hetzner.ds_v1_catalog import (
    CatalogRefresh,
    catalog_exists,
    load_or_discover_catalog,
)
from runner.nodes.hetzner.ds_v1_metadata import metadata_path_for_v1
from runner.nodes.hetzner.ds_v1_storage import (
    cached_parquet_file,
    list_parquet_files,
    prune_metadata_cache,
    prune_parquet_cache,
)
from runner.nodes.hetzner.ds_v2_rows import cached_remote_file


AUDIO_COLUMN = "audio_bytes"
PARQUET_LOOKAHEAD = 2
DOWNLOAD_WORKERS = 2


@dataclass(frozen=True)
class DsV1ParquetRow:
    remote_parquet_path: str
    row_index: int
    values: dict[str, Any]


@dataclass(frozen=True)
class DsV1ParquetRows:
    remote_paths: list[str]
    host: str
    cache_dir: Path
    retries: int
    row_offset: int
    row_limit: int
    check_cancel: Callable[[], None]
    refresh: CatalogRefresh | None

    def __iter__(self) -> Iterator[DsV1ParquetRow]:
        skipped = 0
        emitted = 0
        prefetcher = ParquetPrefetcher(
            self.remote_paths,
            self.host,
            self.cache_dir,
            self.retries,
        )
        try:
            for source_index, remote_path in enumerate(sorted(self.remote_paths)):
                if emitted >= self.row_limit:
                    return
                self.check_cancel()
                local_path = prefetcher.warm(source_index)
                prefetcher.schedule(source_index + 1)
                for row_index, values in iter_parquet_rows(local_path):
                    self.check_cancel()
                    if skipped < self.row_offset:
                        skipped += 1
                        continue
                    yield DsV1ParquetRow(remote_path, row_index, values)
                    emitted += 1
                    if emitted >= self.row_limit:
                        return
                prune_parquet_cache(self.cache_dir)
                prune_metadata_cache(self.cache_dir)
        finally:
            prefetcher.close()
            if self.refresh is not None:
                self.refresh.close()
            prune_parquet_cache(self.cache_dir)
            prune_metadata_cache(self.cache_dir)


class ParquetPrefetcher:
    def __init__(
        self,
        remote_paths: list[str],
        host: str,
        cache_dir: Path,
        retries: int,
    ):
        self._remote_paths = sorted(remote_paths)
        self._host = host
        self._cache_dir = cache_dir
        self._retries = retries
        self._futures: dict[str, Future[Path]] = {}
        self._executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS, thread_name_prefix="ds-v1-sftp")

    def schedule(self, source_index: int) -> None:
        for remote_path in prefetch_window(self._remote_paths, source_index):
            if remote_path not in self._futures:
                self._futures[remote_path] = self._executor.submit(
                    _prefetch_source,
                    self._host,
                    remote_path,
                    self._cache_dir,
                    self._retries,
                )

    def warm(self, source_index: int) -> Path:
        self.schedule(source_index)
        window = prefetch_window(self._remote_paths, source_index)
        local_paths = [self._futures[remote_path].result() for remote_path in window]
        return local_paths[0]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def prefetch_window(remote_paths: list[str], source_index: int) -> list[str]:
    return remote_paths[source_index:source_index + PARQUET_LOOKAHEAD + 1]


def _prefetch_source(host: str, remote_path: str, cache_dir: Path, retries: int) -> Path:
    local_path = cached_parquet_file(host, remote_path, cache_dir, retries)
    metadata_path = cached_remote_file(host, metadata_path_for_v1(remote_path), cache_dir, retries)
    metadata_path.touch()
    return local_path


def load_parquet_rows(
    host: str,
    cache_dir: Path,
    retries: int,
    row_offset: int,
    row_limit: int,
    check_cancel: Callable[[], None],
) -> DsV1ParquetRows:
    cached = catalog_exists(cache_dir, host)
    remote_paths = load_or_discover_catalog(
        cache_dir,
        host,
        lambda: list_parquet_files(host, retries),
    )
    refresh = CatalogRefresh(cache_dir, host, retries) if cached else None
    return DsV1ParquetRows(
        remote_paths,
        host,
        cache_dir,
        retries,
        row_offset,
        row_limit,
        check_cancel,
        refresh,
    )


def iter_parquet_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    if AUDIO_COLUMN not in set(parquet.schema.names):
        raise ValueError(f"Parquet file has no {AUDIO_COLUMN} column: {path}")
    row_index = 0
    for batch in parquet.iter_batches(batch_size=1):
        for row in batch.to_pylist():
            yield row_index, row
            row_index += 1
