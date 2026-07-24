from __future__ import annotations

import json
import multiprocessing
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from runner.nodes.hetzner.ds_v1_storage import list_parquet_files
from runner.nodes.models import stable_id


@dataclass(frozen=True)
class DsV1Catalog:
    host: str
    remote_paths: list[str]


class CatalogRefresh:
    def __init__(self, cache_dir: Path, host: str, retries: int):
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(
            target=_refresh_catalog,
            args=(cache_dir, host, retries),
            daemon=True,
        )
        self._process.start()

    def close(self) -> None:
        if not self._process.is_alive():
            self._process.join()
            return
        os.killpg(self._process.pid, signal.SIGTERM)
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()


def load_or_discover_catalog(
    cache_dir: Path,
    host: str,
    discover: Callable[[], list[str]],
) -> list[str]:
    path = _catalog_path(cache_dir, host)
    if path.exists():
        catalog = DsV1Catalog(**json.loads(path.read_text(encoding="utf-8")))
        if catalog.host != host or not catalog.remote_paths:
            raise ValueError(f"Invalid ds_v1 catalog: {path}")
        return sorted(catalog.remote_paths)
    remote_paths = sorted(discover())
    write_catalog(cache_dir, host, remote_paths)
    return remote_paths


def catalog_exists(cache_dir: Path, host: str) -> bool:
    return _catalog_path(cache_dir, host).exists()


def write_catalog(cache_dir: Path, host: str, remote_paths: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _catalog_path(cache_dir, host)
    temporary = path.with_suffix(".tmp")
    payload = DsV1Catalog(host, sorted(remote_paths))
    temporary.write_text(
        json.dumps({"host": payload.host, "remote_paths": payload.remote_paths}),
        encoding="utf-8",
    )
    temporary.replace(path)


def _catalog_path(cache_dir: Path, host: str) -> Path:
    key = stable_id("ds_v1_catalog", host)
    return cache_dir / f"{key}.json"


def _refresh_catalog(cache_dir: Path, host: str, retries: int) -> None:
    os.setsid()
    write_catalog(cache_dir, host, list_parquet_files(host, retries))
