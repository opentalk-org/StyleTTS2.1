from __future__ import annotations

import io
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from runner.nodes.hetzner.ds_v2_alignment import (
    alignment_from_timestamps,
    alignment_window,
)
from runner.nodes.models import Audio, AudioSegment, stable_id


TEXT_COLUMNS = ("text_src", "text_whisper", "text_parakeet", "text_canary")
TRANSCRIPT_SEGMENTS = (
    ("src", "text_src"),
    ("whisper", "text_whisper"),
    ("parakeet", "text_parakeet"),
    ("canary", "text_canary"),
)


@dataclass(frozen=True)
class DsV2AudioOptions:
    host: str
    remote_parquet_path: str
    text_column: str
    name_prefix: str


def audio_from_row(row: dict[str, Any], options: DsV2AudioOptions, row_index: int, voice_id: UUID | None) -> Audio:
    wav_bytes = _required_bytes(row["audio"], row_index)
    info = _wav_info(wav_bytes)
    duration = _float_or_none(row["duration"]) or info["duration"]
    sample_rate = int(info["sample_rate"])
    channels = int(info["channels"])
    source_key = f"{options.host}:{options.remote_parquet_path}:{row_index}"
    audio_file_id = uuid5(NAMESPACE_URL, source_key)
    text = _text(row, options.text_column)
    score = _float_or_none(row["mos_score"])
    name = _audio_name(options.name_prefix, row, row_index)
    audio_id = stable_id("hetzner_ds_v2_audio", options.remote_parquet_path, row_index)
    segments = _transcript_segments(
        row, options, row_index, audio_file_id, name, duration, sample_rate, channels, score, voice_id
    )
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
        lineage_id=stable_id("hetzner_ds_v2_audio_lineage", options.remote_parquet_path, row_index),
        metadata=_audio_metadata(row, options, row_index, sample_rate, channels, duration, score, text, voice_id),
        byte_length=len(wav_bytes),
        virtual=False,
        segments=segments,
    )


def speaker_name(row: dict[str, Any]) -> str | None:
    return _string_or_none(row["speaker_id"])


def _transcript_segments(
    row: dict[str, Any],
    options: DsV2AudioOptions,
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
        _transcript_segment(
            row, options, row_index, audio_file_id, name, duration, sample_rate,
            channels, score, voice_id, source, column,
        )
        for source, column in TRANSCRIPT_SEGMENTS
        if _string_or_none(row[column])
    ]
    if segments:
        return segments
    return [
        _transcript_segment(
            row, options, row_index, audio_file_id, name, duration, sample_rate,
            channels, score, voice_id, "empty", options.text_column,
        )
    ]


def _transcript_segment(
    row: dict[str, Any],
    options: DsV2AudioOptions,
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
    text = _string_or_none(row[column]) or ""
    timestamps = _json_or_text(row["text_timestamps"]) if column == "text_parakeet" else None
    alignment = None
    if timestamps is not None:
        alignment = alignment_from_timestamps(timestamps, text, alignment_window(row, row_index))
    remote_path = options.remote_parquet_path
    return AudioSegment(
        source_audio_id=audio_file_id,
        name=name,
        start=0.0,
        end=duration,
        sample_rate=sample_rate,
        channels=channels,
        text=text,
        phon="",
        id=stable_id("hetzner_ds_v2_segment", remote_path, row_index, source),
        lineage_id=stable_id("hetzner_ds_v2_segment_lineage", remote_path, row_index, source),
        segment_id=stable_id("hetzner_ds_v2_segment_entry", remote_path, row_index, source),
        speaker=speaker_name(row),
        voice_id=voice_id,
        confidence=score,
        metadata={
            "type_": source,
            "model": source,
            "text_column": column,
            "preferred_text_column": options.text_column,
            "text_timestamps": timestamps,
        },
        alignment=alignment,
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
    options: DsV2AudioOptions,
    row_index: int,
    sample_rate: int,
    channels: int,
    duration: float,
    score: float | None,
    text: str,
    voice_id: UUID | None,
) -> dict[str, Any]:
    speaker = speaker_name(row)
    return {
        "source": "hetzner_ds_v2",
        "source_host": options.host,
        "source_parquet_path": options.remote_parquet_path,
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
        "text_column": options.text_column,
        "text_src": _string_or_none(row["text_src"]),
        "text_parakeet": _string_or_none(row["text_parakeet"]),
        "text_whisper": _string_or_none(row["text_whisper"]),
        "text_canary": _string_or_none(row["text_canary"]),
        "text_timestamps": _json_or_text(row["text_timestamps"]),
        "audio_path": _string_or_none(row["audio_path"]),
        "parquet_filename": _string_or_none(row["parquet_filename"]),
        "filename": _string_or_none(row["filename"]),
        "src_type": _string_or_none(row["src_type"]),
        "src": _string_or_none(row["src"]),
        "source_metadata": _json_or_text(row["metadata"]),
        "chunk_index": _int_or_none(row["chunk_index"]),
        "chunk_start": _float_or_none(row["chunk_start"]),
        "chunk_end": _float_or_none(row["chunk_end"]),
        "speaker_start": _float_or_none(row["speaker_start"]),
        "speaker_end": _float_or_none(row["speaker_end"]),
        "sample_index": _int_or_none(row["sample_index"]),
        "sample_start": _float_or_none(row["sample_start"]),
        "sample_end": _float_or_none(row["sample_end"]),
    }


def _audio_name(prefix: str, row: dict[str, Any], row_index: int) -> str:
    raw = _string_or_none(row["filename"]) or _string_or_none(row["audio_path"]) or f"row_{row_index:06d}.wav"
    stem = Path(raw).stem or f"row_{row_index:06d}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"row_{row_index:06d}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "ds_v2"
    return f"{safe_prefix}_{row_index:06d}_{safe_stem}.wav"


def _text(row: dict[str, Any], preferred_column: str) -> str:
    for column in (preferred_column, *TEXT_COLUMNS):
        value = _string_or_none(row[column])
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
    return None if number != number else number


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
