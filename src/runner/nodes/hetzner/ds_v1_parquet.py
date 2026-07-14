from __future__ import annotations

import importlib
import io
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.hetzner.ds_v1_metadata import load_metadata_index, matching_samples, merge_recording_metadata
from runner.nodes.hetzner.ds_v1_segments import segments_from_samples
from runner.nodes.hetzner.ds_v1_storage import parquet_path
from runner.nodes.models import Audio, stable_id


DEFAULT_PARQUET_PATH = "/home/ds_v1/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z.parquet"
AUDIO_COLUMN = "audio_bytes"
# Structured JSON metadata columns are decoded before storage.
JSON_METADATA_COLUMNS = ("categories_json", "tags_json")
NAME_COLUMNS = ("video_id", "opus_file", "metadata_id", "title")


class HetznerDsV1ParquetAudioSourceSettings(StrictSettings):
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    remote_parquet_path: str = Field(default=DEFAULT_PARQUET_PATH, title="Remote parquet path")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int = Field(default=1, ge=1, title="Rows to import")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = "text_src"
    name_prefix: str = Field(default="ds_v1", title="Audio name prefix")
    cache_download: bool = Field(default=True, title="Cache SFTP download")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")


class HetznerDsV1ParquetAudioSourceNode(Node):
    NODE_TYPE = "HetznerDsV1ParquetAudioSource"
    DESCRIPTION = "Import long ds_v1 recordings from Hetzner, merge their matching ds_v2 metadata, and attach absolute-time transcript and Parakeet alignment segments."
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV1ParquetAudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
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
    cache_dir = Path(context.cache_dir) / "hetzner"
    context.check_cancel()
    metadata_index = load_metadata_index(
        settings.host,
        settings.remote_parquet_path,
        cache_dir,
        settings.download_retries,
        context.check_cancel,
    )
    local_path = parquet_path(settings, context)
    rows = list(_iter_parquet_rows(local_path, settings.row_offset, settings.row_limit))
    items = []
    for absolute_index, row in rows:
        context.check_cancel()
        audio = _audio_from_row(row, settings, absolute_index)
        samples = matching_samples(metadata_index, row, absolute_index)
        items.append(
            replace(
                audio,
                metadata=merge_recording_metadata(audio.metadata, samples, metadata_index.remote_path),
                segments=segments_from_samples(
                    audio,
                    samples,
                    settings.remote_parquet_path,
                    absolute_index,
                    settings.text_column,
                ),
            )
        )
    return items


def _iter_parquet_rows(path: Path, row_offset: int, row_limit: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to import ds_v1 parquet files") from error

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    if AUDIO_COLUMN not in available:
        raise ValueError(f"Parquet file has no {AUDIO_COLUMN} column: {path}")
    # Import all metadata: read every column, not a fixed subset. ds_v1 rows are
    # large (long recordings), so keep the batch small to bound memory.
    seen = 0
    emitted = 0
    for batch in parquet.iter_batches(batch_size=max(1, min(4, row_limit))):
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
    opus_bytes = _required_bytes(row.get(AUDIO_COLUMN), row_index)
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
    # Import all metadata columns from the parquet row (everything except the
    # raw audio payload), decoding JSON blobs and normalizing empty strings.
    metadata: dict[str, Any] = {}
    for column, value in row.items():
        if column == AUDIO_COLUMN:
            continue
        if column in JSON_METADATA_COLUMNS:
            metadata[column] = _json_or_text(value)
        else:
            metadata[column] = _scalar_or_none(value)
    # Provenance and decoded-audio properties. Keys are chosen to not clobber
    # any imported column (the parquet already carries its own `duration`).
    metadata.update(
        {
            "source": "hetzner_ds_v1",
            "source_host": settings.host,
            "source_parquet_path": settings.remote_parquet_path,
            "source_row_index": row_index,
            "source_format": "opus",
            "source_byte_length": source_byte_length,
            "sample_rate": sample_rate,
            "channels": channels,
            "audio_duration_sec": duration,
        }
    )
    return metadata


def _required_bytes(value: Any, row_index: int) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise ValueError(f"ds_v1 row {row_index} has no {AUDIO_COLUMN} bytes")


def _audio_name(prefix: str, row: dict[str, Any], row_index: int) -> str:
    raw = next((value for column in NAME_COLUMNS if (value := _string_or_none(row.get(column)))), None)
    raw = raw or f"row_{row_index:06d}"
    stem = Path(raw).stem or f"row_{row_index:06d}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"row_{row_index:06d}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "ds_v1"
    return f"{safe_prefix}_{row_index:06d}_{safe_stem}.wav"


def _json_or_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, (str, bytes)):
        return value
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value if value else None
    return value


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    return text if text else None
