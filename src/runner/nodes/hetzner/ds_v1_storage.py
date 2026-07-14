from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runner.nodes.models import stable_id

if TYPE_CHECKING:
    from runner.nodes.hetzner.ds_v1_parquet import HetznerDsV1ParquetAudioSourceSettings


def parquet_path(settings: HetznerDsV1ParquetAudioSourceSettings, context: Any) -> Path:
    cache_dir = Path(context.cache_dir) / "hetzner"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / cache_name(settings.remote_parquet_path)
    if cached.exists() and settings.cache_download:
        return cached
    target = cached if settings.cache_download else cache_dir / f"download_{cache_name(settings.remote_parquet_path)}"
    _download_sftp_file(settings.host, settings.remote_parquet_path, target, settings.download_retries)
    return target


def cache_name(remote_path: str) -> str:
    source = Path(remote_path).name or "ds_v1.parquet"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
    digest = stable_id("remote", remote_path).removeprefix("remote_")
    return f"{digest}_{safe}"


def _download_sftp_file(host: str, remote_path: str, target: Path, retries: int) -> None:
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    last_detail = ""
    for attempt in range(1, retries + 1):
        tmp.unlink(missing_ok=True)
        batch = f"get {remote_path} {tmp}\n"
        result = subprocess.run(
            [
                "sftp",
                "-q",
                "-oBatchMode=yes",
                "-oConnectTimeout=30",
                "-oConnectionAttempts=3",
                "-oServerAliveInterval=15",
                "-b",
                "-",
                host,
            ],
            input=batch,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(target)
            return
        last_detail = _sftp_error_detail(result, tmp)
        tmp.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(f"SFTP download failed after {retries} attempt(s) for {host}:{remote_path}: {last_detail}")


def _sftp_error_detail(result: subprocess.CompletedProcess[str], tmp: Path) -> str:
    parts = [
        f"exit={result.returncode}",
        f"tmp_exists={tmp.exists()}",
        f"tmp_bytes={tmp.stat().st_size if tmp.exists() else 0}",
        result.stderr.strip(),
        result.stdout.strip(),
    ]
    return " | ".join(part for part in parts if part)
