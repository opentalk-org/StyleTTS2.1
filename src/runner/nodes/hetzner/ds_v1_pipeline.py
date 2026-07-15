from __future__ import annotations

import json
import re
import shutil
import time
from collections import deque
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from runflow.core.settings import StrictSettings
from runner.nodes.hetzner.ds_v1_conversion import ConvertedWav, OpusConversionPool
from runner.nodes.hetzner.ds_v1_metadata import (
    DsV2MetadataIndex,
    load_metadata_index,
    matching_samples,
    merge_recording_metadata,
)
from runner.nodes.hetzner.ds_v1_segments import segments_from_samples
from runner.nodes.hetzner.ds_v1_sources import AUDIO_COLUMN, DsV1ParquetRow, load_parquet_rows
from runner.nodes.models import Audio, stable_id


JSON_METADATA_COLUMNS = ("categories_json", "tags_json")
NAME_COLUMNS = ("video_id", "opus_file", "metadata_id", "title")
CONVERSION_WORKERS = 4
CONVERSION_LOOKAHEAD = CONVERSION_WORKERS * 4


class HetznerDsV1ParquetAudioSourceSettings(StrictSettings):
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int = Field(default=1, ge=1, title="Rows to import")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = "text_src"
    name_prefix: str = Field(default="ds_v1", title="Audio name prefix")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")


@dataclass(frozen=True)
class PendingConversion:
    source: DsV1ParquetRow
    source_byte_length: int
    opus_path: Path
    future: Future[ConvertedWav]


class DsV1AudioPipeline:
    def __init__(self, settings: HetznerDsV1ParquetAudioSourceSettings, context: Any, node_id: str):
        self._settings = settings
        self._context = context
        self._temp_dir = context.node_dir(node_id) / "conversion"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        rows = load_parquet_rows(
            settings.host,
            Path(context.cache_dir) / "hetzner",
            settings.download_retries,
            settings.row_offset,
            settings.row_limit,
            context.check_cancel,
        )
        self._rows = iter(rows)
        self._converter = OpusConversionPool(self._temp_dir, CONVERSION_WORKERS)
        self._pending: deque[PendingConversion] = deque()
        self._metadata_indexes: dict[str, DsV2MetadataIndex] = {}
        self._submitted = 0
        self._emitted = 0
        self._audio_seconds = 0.0
        self._started_at = time.perf_counter()
        self._fill()
        self._warm_conversion_buffer()

    @property
    def remaining(self) -> int:
        return self._settings.row_limit - self._emitted

    @property
    def realtime_factor(self) -> float:
        elapsed = time.perf_counter() - self._started_at
        return self._audio_seconds / elapsed if elapsed > 0 else 0.0

    def next_audio(self) -> Audio | None:
        if not self._pending:
            return None
        self._context.check_cancel()
        pending = self._pending.popleft()
        converted = pending.future.result()
        audio = _audio_from_conversion(
            pending.source,
            pending.source_byte_length,
            converted,
            self._settings,
        )
        index = self._metadata_index(pending.source.remote_parquet_path)
        samples = matching_samples(index, pending.source.values, pending.source.row_index)
        enriched = replace(
            audio,
            metadata=merge_recording_metadata(audio.metadata, samples, index.remote_path),
            segments=segments_from_samples(
                audio,
                samples,
                pending.source.remote_parquet_path,
                pending.source.row_index,
                self._settings.text_column,
            ),
        )
        pending.opus_path.unlink(missing_ok=True)
        converted.wav_path.unlink(missing_ok=True)
        self._emitted += 1
        self._audio_seconds += enriched.duration
        self._fill()
        return enriched

    def close(self) -> None:
        close_rows = getattr(self._rows, "close", None)
        if close_rows is not None:
            close_rows()
        self._converter.close()
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _fill(self) -> None:
        while len(self._pending) < CONVERSION_LOOKAHEAD:
            self._context.check_cancel()
            try:
                source = next(self._rows)
            except StopIteration:
                return
            opus_bytes = _required_bytes(source.values[AUDIO_COLUMN], source.row_index)
            source_values = {key: value for key, value in source.values.items() if key != AUDIO_COLUMN}
            prepared = replace(source, values=source_values)
            opus_path = self._temp_dir / f"{self._submitted:08d}.opus"
            opus_path.write_bytes(opus_bytes)
            future = self._converter.submit(opus_path, self._submitted)
            self._pending.append(PendingConversion(prepared, len(opus_bytes), opus_path, future))
            self._submitted += 1

    def _warm_conversion_buffer(self) -> None:
        unfinished = {pending.future for pending in self._pending}
        while unfinished:
            self._context.check_cancel()
            finished, unfinished = wait(unfinished, timeout=0.25, return_when=FIRST_EXCEPTION)
            for future in finished:
                future.result()

    def _metadata_index(self, remote_parquet_path: str) -> DsV2MetadataIndex:
        if remote_parquet_path not in self._metadata_indexes:
            self._metadata_indexes[remote_parquet_path] = load_metadata_index(
                self._settings.host,
                remote_parquet_path,
                Path(self._context.cache_dir) / "hetzner",
                self._settings.download_retries,
                self._context.check_cancel,
            )
        return self._metadata_indexes[remote_parquet_path]


def _audio_from_conversion(
    source: DsV1ParquetRow,
    source_byte_length: int,
    converted: ConvertedWav,
    settings: HetznerDsV1ParquetAudioSourceSettings,
) -> Audio:
    wav_bytes = converted.wav_path.read_bytes()
    remote_path = source.remote_parquet_path
    row_index = source.row_index
    audio_file_id = uuid5(NAMESPACE_URL, f"{settings.host}:{remote_path}:{row_index}")
    name = _audio_name(settings.name_prefix, source.values, row_index)
    return Audio(
        audio_file_id=audio_file_id,
        name=name,
        data=wav_bytes,
        sample_rate=converted.sample_rate,
        channels=converted.channels,
        start=0.0,
        end=converted.duration,
        confidence=1.0,
        id=stable_id("hetzner_ds_v1_audio", remote_path, row_index),
        lineage_id=stable_id("hetzner_ds_v1_audio_lineage", remote_path, row_index),
        metadata=_audio_metadata(
            source.values,
            settings,
            remote_path,
            row_index,
            converted,
            source_byte_length,
        ),
        byte_length=len(wav_bytes),
        virtual=False,
        segments=[],
    )


def _audio_metadata(
    row: dict[str, Any],
    settings: HetznerDsV1ParquetAudioSourceSettings,
    remote_path: str,
    row_index: int,
    converted: ConvertedWav,
    source_byte_length: int,
) -> dict[str, Any]:
    metadata = {
        column: _json_or_text(value) if column in JSON_METADATA_COLUMNS else _scalar_or_none(value)
        for column, value in row.items()
    }
    metadata.update(
        {
            "source": "hetzner_ds_v1",
            "source_host": settings.host,
            "source_parquet_path": remote_path,
            "source_row_index": row_index,
            "source_format": "opus",
            "source_byte_length": source_byte_length,
            "sample_rate": converted.sample_rate,
            "channels": converted.channels,
            "audio_duration_sec": converted.duration,
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
    if value is None or isinstance(value, (dict, list)):
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
    if value is None or isinstance(value, (bool, int, float)):
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
