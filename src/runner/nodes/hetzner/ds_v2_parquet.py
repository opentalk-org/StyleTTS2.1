from __future__ import annotations

import io
import json
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioSegment, stable_id
from shared.db import database_session
from shared.db.voices.models import Voice


DEFAULT_PARQUET_PATH = "/home/ds_v2/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet"
TEXT_COLUMNS = ("text_src", "text_whisper", "text_parakeet", "text_canary")
TRANSCRIPT_SEGMENTS = (
    ("src", "text_src"),
    ("whisper", "text_whisper"),
    ("parakeet", "text_parakeet"),
    ("canary", "text_canary"),
)
PARQUET_COLUMNS = (
    "duration",
    "chunk_index",
    "chunk_start",
    "chunk_end",
    "speaker_start",
    "speaker_end",
    "sample_index",
    "sample_start",
    "sample_end",
    "text_parakeet",
    "text_timestamps",
    "text_whisper",
    "text_canary",
    "text_src",
    "mos_score",
    "audio_path",
    "parquet_filename",
    "filename",
    "src_type",
    "src",
    "metadata",
    "speaker_id",
    "audio",
)


class HetznerDsV2ParquetAudioSourceSettings(StrictSettings):
    source: Literal["sftp", "local"] = Field(default="sftp", title="Source")
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    remote_parquet_path: str = Field(default=DEFAULT_PARQUET_PATH, title="Remote parquet path")
    local_parquet_path: str = Field(default="", title="Local parquet path")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int = Field(default=1, ge=1, title="Rows to import")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = Field(default="text_src", title="Transcript column")
    name_prefix: str = Field(default="ds_v2", title="Audio name prefix")
    cache_download: bool = Field(default=True, title="Cache SFTP download")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")
    create_voices: bool = Field(default=True, title="Create voices")


class HetznerDsV2ParquetAudioSourceNode(Node):
    NODE_TYPE = "HetznerDsV2ParquetAudioSource"
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV2ParquetAudioSourceSettings
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


def _load_audio_items(settings: HetznerDsV2ParquetAudioSourceSettings, context: Any) -> list[Audio]:
    parquet_path = _local_parquet_path(settings, context)
    rows = list(_iter_parquet_rows(parquet_path, settings.row_offset, settings.row_limit))
    voice_ids = _voice_ids_for_rows(settings, [row for _, row in rows])
    return [
        _audio_from_row(row, settings, absolute_index, voice_ids.get(_speaker_name(row)))
        for absolute_index, row in rows
    ]


def _local_parquet_path(settings: HetznerDsV2ParquetAudioSourceSettings, context: Any) -> Path:
    if settings.source == "local":
        if not settings.local_parquet_path:
            raise ValueError("local_parquet_path is required when source is local")
        return Path(settings.local_parquet_path).expanduser()
    cache_dir = Path(context.cache_dir) / "hetzner"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _cache_name(settings.remote_parquet_path)
    if cached.exists() and settings.cache_download:
        return cached
    target = cached if settings.cache_download else cache_dir / f"download_{_cache_name(settings.remote_parquet_path)}"
    _download_sftp_file(settings.host, settings.remote_parquet_path, target, settings.download_retries)
    return target


def _cache_name(remote_path: str) -> str:
    source = Path(remote_path).name or "ds_v2.parquet"
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
        raise RuntimeError("pyarrow is required to import ds_v2 parquet files") from error

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    if "audio" not in available:
        raise ValueError(f"Parquet file has no audio column: {path}")
    columns = [name for name in PARQUET_COLUMNS if name in available]
    seen = 0
    emitted = 0
    for batch in parquet.iter_batches(batch_size=max(1, min(64, row_limit)), columns=columns):
        for row in batch.to_pylist():
            absolute_index = seen
            seen += 1
            if absolute_index < row_offset:
                continue
            yield absolute_index, row
            emitted += 1
            if emitted >= row_limit:
                return


def _audio_from_row(row: dict[str, Any], settings: HetznerDsV2ParquetAudioSourceSettings, row_index: int, voice_id: UUID | None) -> Audio:
    wav_bytes = _required_bytes(row.get("audio"), row_index)
    info = _wav_info(wav_bytes)
    duration = _float_or_none(row.get("duration")) or info["duration"]
    sample_rate = info["sample_rate"]
    channels = info["channels"]
    audio_file_id = uuid5(NAMESPACE_URL, f"{settings.host}:{settings.remote_parquet_path}:{row_index}")
    text = _text(row, settings.text_column)
    score = _float_or_none(row.get("mos_score"))
    name = _audio_name(settings.name_prefix, row, row_index)
    audio_id = stable_id("hetzner_ds_v2_audio", settings.remote_parquet_path, row_index)
    segments = _transcript_segments(row, settings, row_index, audio_file_id, name, duration, sample_rate, channels, score, voice_id)
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
        lineage_id=stable_id("hetzner_ds_v2_audio_lineage", settings.remote_parquet_path, row_index),
        metadata=_audio_metadata(row, settings, row_index, sample_rate, channels, duration, score, text, voice_id),
        byte_length=len(wav_bytes),
        virtual=False,
        segments=segments,
    )


def _transcript_segments(
    row: dict[str, Any],
    settings: HetznerDsV2ParquetAudioSourceSettings,
    row_index: int,
    audio_file_id: UUID,
    name: str,
    duration: float,
    sample_rate: int,
    channels: int,
    score: float | None,
    voice_id: UUID | None,
) -> list[AudioSegment]:
    segments = [
        _transcript_segment(row, settings, row_index, audio_file_id, name, duration, sample_rate, channels, score, voice_id, source, column)
        for source, column in TRANSCRIPT_SEGMENTS
        if _string_or_none(row.get(column))
    ]
    if segments:
        return segments
    return [_transcript_segment(row, settings, row_index, audio_file_id, name, duration, sample_rate, channels, score, voice_id, "empty", settings.text_column)]


def _transcript_segment(
    row: dict[str, Any],
    settings: HetznerDsV2ParquetAudioSourceSettings,
    row_index: int,
    audio_file_id: UUID,
    name: str,
    duration: float,
    sample_rate: int,
    channels: int,
    score: float | None,
    voice_id: UUID | None,
    source: str,
    column: str,
) -> AudioSegment:
    text = _string_or_none(row.get(column)) or ""
    return AudioSegment(
        source_audio_id=audio_file_id,
        name=name,
        start=0.0,
        end=duration,
        sample_rate=sample_rate,
        channels=channels,
        text=text,
        phon="",
        id=stable_id("hetzner_ds_v2_segment", settings.remote_parquet_path, row_index, source),
        lineage_id=stable_id("hetzner_ds_v2_segment_lineage", settings.remote_parquet_path, row_index, source),
        segment_id=stable_id("hetzner_ds_v2_segment_entry", settings.remote_parquet_path, row_index, source),
        speaker=_speaker_name(row),
        voice_id=voice_id,
        confidence=score,
        metadata={
            "type_": source,
            "model": source,
            "text_column": column,
            "preferred_text_column": settings.text_column,
            "text_timestamps": _json_or_text(row.get("text_timestamps")) if column == "text_parakeet" else None,
        },
    )


def _required_bytes(value: Any, row_index: int) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise ValueError(f"ds_v2 row {row_index} has no audio bytes")


def _wav_info(wav_bytes: bytes) -> dict[str, float | int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        return {
            "sample_rate": sample_rate,
            "channels": wav_file.getnchannels(),
            "duration": frames / float(sample_rate) if sample_rate > 0 else 0.0,
        }


def _audio_metadata(
    row: dict[str, Any],
    settings: HetznerDsV2ParquetAudioSourceSettings,
    row_index: int,
    sample_rate: int,
    channels: int,
    duration: float,
    score: float | None,
    text: str,
    voice_id: UUID | None,
) -> dict[str, Any]:
    speaker = _speaker_name(row)
    return {
        "source": "hetzner_ds_v2",
        "source_host": settings.host,
        "source_parquet_path": settings.remote_parquet_path,
        "source_row_index": row_index,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "score": score,
        "mos_score": score,
        "speaker": speaker or "",
        "speaker_id": speaker,
        "voice_id": str(voice_id) if voice_id is not None else None,
        "text": text,
        "text_column": settings.text_column,
        "text_src": _string_or_none(row.get("text_src")),
        "text_parakeet": _string_or_none(row.get("text_parakeet")),
        "text_whisper": _string_or_none(row.get("text_whisper")),
        "text_canary": _string_or_none(row.get("text_canary")),
        "text_timestamps": _json_or_text(row.get("text_timestamps")),
        "audio_path": _string_or_none(row.get("audio_path")),
        "parquet_filename": _string_or_none(row.get("parquet_filename")),
        "filename": _string_or_none(row.get("filename")),
        "src_type": _string_or_none(row.get("src_type")),
        "src": _string_or_none(row.get("src")),
        "source_metadata": _json_or_text(row.get("metadata")),
        "chunk_index": _int_or_none(row.get("chunk_index")),
        "chunk_start": _float_or_none(row.get("chunk_start")),
        "chunk_end": _float_or_none(row.get("chunk_end")),
        "speaker_start": _float_or_none(row.get("speaker_start")),
        "speaker_end": _float_or_none(row.get("speaker_end")),
        "sample_index": _int_or_none(row.get("sample_index")),
        "sample_start": _float_or_none(row.get("sample_start")),
        "sample_end": _float_or_none(row.get("sample_end")),
    }


def _voice_ids_for_rows(settings: HetznerDsV2ParquetAudioSourceSettings, rows: list[dict[str, Any]]) -> dict[str, UUID]:
    if not settings.create_voices:
        return {}
    names = sorted({speaker for row in rows if (speaker := _speaker_name(row))})
    if not names:
        return {}
    with database_session() as session:
        session.execute(
            insert(Voice)
            .values([{"name": name} for name in names])
            .on_conflict_do_nothing(index_elements=["name"])
        )
        session.commit()
        voices = session.execute(select(Voice).where(Voice.name.in_(names))).scalars().all()
        return {voice.name: voice.id for voice in voices}


def _speaker_name(row: dict[str, Any]) -> str | None:
    return _string_or_none(row.get("speaker_id"))


def _audio_name(prefix: str, row: dict[str, Any], row_index: int) -> str:
    raw = _string_or_none(row.get("filename")) or _string_or_none(row.get("audio_path")) or f"row_{row_index:06d}.wav"
    stem = Path(raw).stem or f"row_{row_index:06d}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"row_{row_index:06d}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "ds_v2"
    return f"{safe_prefix}_{row_index:06d}_{safe_stem}.wav"


def _text(row: dict[str, Any], preferred_column: str) -> str:
    for column in (preferred_column, *TEXT_COLUMNS):
        value = _string_or_none(row.get(column))
        if value:
            return value
    return ""


def _json_or_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if number != number:
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
