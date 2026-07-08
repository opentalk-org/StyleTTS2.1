from __future__ import annotations

import importlib
import io
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AUDIO
from runner.nodes.models import Audio, stable_id


DEFAULT_PARQUET_PATH = "/home/ds_v1/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z.parquet"
PARQUET_COLUMNS = ("filename", "opus")


class HetznerDsV1ParquetAudioSourceSettings(StrictSettings):
    source: Literal["sftp", "local"] = Field(default="sftp", title="Source")
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    remote_parquet_path: str = Field(default=DEFAULT_PARQUET_PATH, title="Remote parquet path")
    local_parquet_path: str = Field(default="", title="Local parquet path")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int = Field(default=1, ge=1, title="Rows to import")
    name_prefix: str = Field(default="ds_v1", title="Audio name prefix")
    cache_download: bool = Field(default=True, title="Cache SFTP download")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")


class HetznerDsV1ParquetAudioSourceNode(Node):
    NODE_TYPE = "HetznerDsV1ParquetAudioSource"
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV1ParquetAudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._items: list[Audio] | None = None
        self._cursor = 0

    def remaining_items(self, context: Any) -> int:
        if self._items is None:
            return self.settings.row_limit
        return len(self._items) - self._cursor

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        if self._items is None:
            self._items = list(_load_audio_items(self.settings, context))
        end = self._cursor + self.runtime.queue_max_size
        items = self._items[self._cursor:end]
        self._cursor += len(items)
        return [{"audio": item} for item in items]


def _load_audio_items(settings: HetznerDsV1ParquetAudioSourceSettings, context: Any) -> list[Audio]:
    parquet_path = _local_parquet_path(settings, context)
    rows = list(_iter_parquet_rows(parquet_path, settings.row_offset, settings.row_limit))
    return [_audio_from_row(row, settings, absolute_index) for absolute_index, row in rows]


def _local_parquet_path(settings: HetznerDsV1ParquetAudioSourceSettings, context: Any) -> Path:
    if settings.source == "local":
        if not settings.local_parquet_path:
            raise ValueError("local_parquet_path is required when source is local")
        return Path(settings.local_parquet_path).expanduser()
    cache_dir = Path(context.cache_dir) / "hetzner" / "ds_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _cache_name(settings.remote_parquet_path)
    if cached.exists() and settings.cache_download:
        return cached
    target = cached if settings.cache_download else cache_dir / f"download_{_cache_name(settings.remote_parquet_path)}"
    _download_sftp_file(settings.host, settings.remote_parquet_path, target, settings.download_retries)
    return target


def _cache_name(remote_path: str) -> str:
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


def _iter_parquet_rows(path: Path, row_offset: int, row_limit: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to import ds_v1 parquet files") from error

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    if "opus" not in available:
        raise ValueError(f"Parquet file has no opus column: {path}")
    columns = [name for name in PARQUET_COLUMNS if name in available]
    seen = 0
    emitted = 0
    for batch in parquet.iter_batches(batch_size=max(1, min(16, row_limit)), columns=columns):
        for row in batch.to_pylist():
            absolute_index = seen
            seen += 1
            if absolute_index < row_offset:
                continue
            yield absolute_index, row
            emitted += 1
            if emitted >= row_limit:
                return


def _audio_from_row(row: dict[str, Any], settings: HetznerDsV1ParquetAudioSourceSettings, row_index: int) -> Audio:
    opus_bytes = _required_bytes(row.get("opus"), row_index)
    wav_bytes, info = _decode_opus_to_wav(opus_bytes)
    duration = info["duration"]
    sample_rate = info["sample_rate"]
    channels = info["channels"]
    audio_file_id = uuid5(NAMESPACE_URL, f"{settings.host}:{settings.remote_parquet_path}:{row_index}")
    name = _audio_name(settings.name_prefix, row, row_index)
    audio_id = stable_id("hetzner_ds_v1_audio", settings.remote_parquet_path, row_index)
    return Audio(
        audio_file_id=audio_file_id,
        name=name,
        data=wav_bytes,
        sample_rate=sample_rate,
        channels=channels,
        start=0.0,
        end=duration,
        confidence=1.0,
        id=audio_id,
        lineage_id=stable_id("hetzner_ds_v1_audio_lineage", settings.remote_parquet_path, row_index),
        metadata=_audio_metadata(row, settings, row_index, sample_rate, channels, duration, len(opus_bytes)),
        byte_length=len(wav_bytes),
        virtual=False,
        segments=[],
    )


def _decode_opus_to_wav(opus_bytes: bytes) -> tuple[bytes, dict[str, float | int]]:
    deps = _audio_deps()
    sf = deps["soundfile"]
    np = deps["numpy"]
    samples, sample_rate = sf.read(io.BytesIO(opus_bytes), always_2d=True, dtype="float32")
    frames = int(samples.shape[0])
    channels = int(samples.shape[1])
    out = io.BytesIO()
    sf.write(out, np.asarray(samples, dtype=np.float32), int(sample_rate), format="WAV", subtype="PCM_16")
    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0
    return out.getvalue(), {"sample_rate": int(sample_rate), "channels": channels, "duration": duration}


def _audio_deps() -> dict[str, Any]:
    modules = {}
    for name in ("numpy", "soundfile"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(f"{name} is required to decode ds_v1 opus audio") from error
    return modules


def _audio_metadata(
    row: dict[str, Any],
    settings: HetznerDsV1ParquetAudioSourceSettings,
    row_index: int,
    sample_rate: int,
    channels: int,
    duration: float,
    source_byte_length: int,
) -> dict[str, Any]:
    return {
        "source": "hetzner_ds_v1",
        "source_host": settings.host,
        "source_parquet_path": settings.remote_parquet_path,
        "source_row_index": row_index,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "filename": _string_or_none(row.get("filename")),
        "source_format": "opus",
        "source_byte_length": source_byte_length,
    }


def _required_bytes(value: Any, row_index: int) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise ValueError(f"ds_v1 row {row_index} has no opus bytes")


def _audio_name(prefix: str, row: dict[str, Any], row_index: int) -> str:
    raw = _string_or_none(row.get("filename")) or f"row_{row_index:06d}.opus"
    stem = Path(raw).stem or f"row_{row_index:06d}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"row_{row_index:06d}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "ds_v1"
    return f"{safe_prefix}_{row_index:06d}_{safe_stem}.wav"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
