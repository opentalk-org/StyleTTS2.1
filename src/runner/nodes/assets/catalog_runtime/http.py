from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

from runner.nodes.assets.credentials import huggingface_auth_headers


USER_AGENT = "StyleTTS-Studio/2.0 (runner catalog download)"

CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_STEP_PERCENT = 10
PROGRESS_STEP_BYTES = 64 * 1024 * 1024

_LOGGER = logging.getLogger(__name__)


def download_url_to_file(url: str, dest: Path, *, error_prefix: str, logger: logging.Logger | None = None) -> None:
    log = logger or _LOGGER
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **huggingface_auth_headers(url)})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            total = _content_length(response)
            log.info("download starting file=%s total_bytes=%s url=%s", dest.name, total or "unknown", url)
            downloaded = _stream_to_file(response, partial, total, dest.name, log)
        partial.replace(dest)
        log.info("download completed file=%s bytes=%s", dest.name, downloaded)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        log.warning("download failed file=%s http_status=%s", dest.name, exc.code)
        raise ValueError(f"{error_prefix}_http_{exc.code}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        log.warning("download failed file=%s error=%s", dest.name, exc)
        raise ValueError(f"{error_prefix}_io_failed") from exc


def download_url_bytes(url: str, *, error_prefix: str, logger: logging.Logger | None = None) -> bytes:
    log = logger or _LOGGER
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **huggingface_auth_headers(url)})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            total = _content_length(response)
            log.info("download starting kind=bytes total_bytes=%s url=%s", total or "unknown", url)
            data = response.read()
        log.info("download completed kind=bytes bytes=%s", len(data))
        return data
    except urllib.error.HTTPError as exc:
        log.warning("download failed kind=bytes http_status=%s", exc.code)
        raise ValueError(f"{error_prefix}_http_{exc.code}") from exc
    except OSError as exc:
        log.warning("download failed kind=bytes error=%s", exc)
        raise ValueError(f"{error_prefix}_io_failed") from exc


def _content_length(response) -> int:
    raw = response.headers.get("Content-Length")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _stream_to_file(response, partial: Path, total: int, name: str, log: logging.Logger) -> int:
    downloaded = 0
    next_mark = PROGRESS_STEP_PERCENT if total else PROGRESS_STEP_BYTES
    with partial.open("wb") as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                if percent >= next_mark:
                    log.info(
                        "download progress file=%s percent=%s downloaded_bytes=%s total_bytes=%s",
                        name, percent, downloaded, total,
                    )
                    next_mark = percent - (percent % PROGRESS_STEP_PERCENT) + PROGRESS_STEP_PERCENT
            elif downloaded >= next_mark:
                log.info("download progress file=%s downloaded_bytes=%s total_bytes=unknown", name, downloaded)
                next_mark += PROGRESS_STEP_BYTES
    return downloaded
