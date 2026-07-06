from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path


USER_AGENT = "StyleTTS-Studio/2.0 (runner catalog download)"


def download_url_to_file(url: str, dest: Path, *, error_prefix: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        partial.replace(dest)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{error_prefix}_http_{exc.code}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{error_prefix}_io_failed") from exc
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise ValueError(error_prefix) from exc


def download_url_bytes(url: str, *, error_prefix: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"{error_prefix}_http_{exc.code}") from exc
    except OSError as exc:
        raise ValueError(f"{error_prefix}_io_failed") from exc
    except Exception as exc:
        raise ValueError(error_prefix) from exc

