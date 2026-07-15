from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path, PurePosixPath

from runner.nodes.models import stable_id


PARQUET_DIRECTORY = PurePosixPath("/home/ds_v1")
MAX_CACHED_PARQUETS = 4
MAX_CACHED_METADATA = 4


def list_parquet_files(host: str, retries: int) -> list[str]:
    command = f"ls -1 {PARQUET_DIRECTORY}/*.parquet\n"
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
            if line.strip().endswith(".parquet") and not line.strip().startswith("sftp>")
        ]
        paths = sorted(
            str(PurePosixPath(name) if name.startswith("/") else PARQUET_DIRECTORY / name)
            for name in names
        )
        if result.returncode == 0 and paths:
            return paths
        last_detail = f"exit={result.returncode} | {result.stderr.strip()} | {result.stdout.strip()}"
        if attempt < retries:
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(
        f"SFTP parquet listing failed after {retries} attempt(s) for {host}:{PARQUET_DIRECTORY}: {last_detail}"
    )


def cached_parquet_file(host: str, remote_path: str, cache_dir: Path, retries: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / cache_name(remote_path)
    if not cached.exists() or cached.stat().st_size == 0:
        _download_sftp_file(host, remote_path, cached, retries)
    cached.touch()
    return cached


def prune_parquet_cache(cache_dir: Path) -> None:
    paths = sorted(
        cache_dir.glob("ds_v1_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths[MAX_CACHED_PARQUETS:]:
        path.unlink()


def prune_metadata_cache(cache_dir: Path) -> None:
    paths = sorted(
        cache_dir.glob("ds_v2_*_processed_metadata.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths[MAX_CACHED_METADATA:]:
        path.unlink()


def cache_name(remote_path: str) -> str:
    source = PurePosixPath(remote_path).name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
    digest = stable_id("remote", remote_path).removeprefix("remote_")
    return f"ds_v1_{digest}_{safe}"


def _download_sftp_file(host: str, remote_path: str, target: Path, retries: int) -> None:
    temporary = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
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
