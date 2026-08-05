import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from imports.stage1.common.hf22_catalog import (
    LocaleSpec,
    SPLITS,
    VALID_SPLITS,
)


BASE_URL = (
    "https://huggingface.co/datasets/"
    "fsicoli/common_voice_22_0/resolve/main"
)
CHUNK_SIZE = 8 * 1024 * 1024
MIN_FREE_BYTES = 48 * 1024**3


class LocaleDeadlineExceeded(RuntimeError):
    pass


class DiskReserveReached(RuntimeError):
    pass


@dataclass(frozen=True)
class ShardTask:
    split: str
    index: int


class HuggingFaceClient:
    def __init__(self, token: str, disk_root: Path) -> None:
        self.disk_root = disk_root
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": "runflow-stage1-common-voice-hf22",
        })

    def catalog_files(
        self,
        directory: Path,
        deadline: float,
    ) -> tuple[Path, Path]:
        stats = directory / "release_stats.py"
        shards = directory / "n_shards.json"
        self.download(
            f"{BASE_URL}/release_stats.py",
            stats,
            deadline,
            enforce_disk_reserve=False,
        )
        self.download(
            f"{BASE_URL}/n_shards.json",
            shards,
            deadline,
            enforce_disk_reserve=False,
        )
        json.loads(shards.read_text(encoding="utf-8"))
        return stats, shards

    def metadata_files(
        self,
        spec: LocaleSpec,
        directory: Path,
        deadline: float,
    ) -> dict[str, Path]:
        paths = {}
        for split in SPLITS:
            path = directory / f"{split}.tsv"
            self.download(
                f"{BASE_URL}/transcript/{spec.hf_locale}/{split}.tsv",
                path,
                deadline,
                enforce_disk_reserve=False,
            )
            paths[split] = path
        return paths

    def shard(
        self,
        spec: LocaleSpec,
        task: ShardTask,
        directory: Path,
        deadline: float,
    ) -> Path:
        filename = f"{spec.hf_locale}_{task.split}_{task.index}.tar"
        path = directory / filename
        url = (
            f"{BASE_URL}/audio/{spec.hf_locale}/{task.split}/{filename}"
        )
        self.download(url, path, deadline, enforce_disk_reserve=True)
        return path

    def download(
        self,
        url: str,
        target: Path,
        deadline: float,
        enforce_disk_reserve: bool,
    ) -> None:
        if target.is_file() and target.stat().st_size > 0:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.parent / f"{target.name}.part"
        last_error: Exception | None = None
        for attempt in range(5):
            _check_deadline(deadline)
            if enforce_disk_reserve:
                _check_disk(self.disk_root)
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(15, 120),
                ) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    mode = "ab" if append else "wb"
                    with partial.open(mode) as output:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            _check_deadline(deadline)
                            if chunk:
                                output.write(chunk)
                assert partial.stat().st_size > 0, f"{url}: empty download"
                partial.replace(target)
                return
            except (
                requests.ConnectionError,
                requests.HTTPError,
                requests.Timeout,
            ) as error:
                last_error = error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LocaleDeadlineExceeded(url) from error
                time.sleep(min(2**attempt, 16, remaining))
        raise RuntimeError(f"{url}: download failed after retries") from last_error


def load_hf_token(repository_root: Path) -> str:
    for line in (repository_root / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            assert token, "HF_TOKEN is empty"
            return token
    raise RuntimeError("HF_TOKEN is missing from .env")


def shard_order(spec: LocaleSpec) -> list[ShardTask]:
    split_orders = {
        split: _spread_indices(spec.shards.count(split))
        for split in VALID_SPLITS
    }
    tasks = []
    position = 0
    while any(position < len(indices) for indices in split_orders.values()):
        for split in VALID_SPLITS:
            indices = split_orders[split]
            if position < len(indices):
                tasks.append(ShardTask(split=split, index=indices[position]))
        position += 1
    return tasks


def _spread_indices(count: int) -> list[int]:
    if count <= 2:
        return list(range(count))
    selected = [0, count - 1]
    remaining = set(range(1, count - 1))
    while remaining:
        choice = max(
            remaining,
            key=lambda index: min(abs(index - item) for item in selected),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise LocaleDeadlineExceeded("locale exceeded its 45-minute deadline")


def _check_disk(root: Path) -> None:
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise DiskReserveReached(
            f"free disk {free / 1024**3:.2f} GiB is below 48 GiB reserve"
        )
