from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from runner.nodes.models import Audio, AudioSegment, SaveResult, stable_id
from shared.db.audio.schemas import ExternalAudioCreate, ExternalAudioLocation


def external_payload(audio: Audio) -> ExternalAudioCreate:
    if audio.data is not None:
        raise ValueError(f"external mode requires metadata-only audio: {audio.audio_file_id}")
    metadata = audio.metadata
    return ExternalAudioCreate(
        id=audio.audio_file_id,
        name=audio.name,
        duration=audio.duration,
        score=_optional_float(metadata["mos_score"]),
        language=str(metadata["language"]) if "language" in metadata and metadata["language"] else None,
        style_prompt=audio.style_prompt,
        voice_prompt=audio.voice_prompt,
        segments=[_segment_dict(segment) for segment in audio.segments],
        metadata=metadata,
        storage_ref=ExternalAudioLocation(
            provider=str(metadata["storage_provider"]),
            host=str(metadata["source_host"]),
            path=str(metadata["source_parquet_path"]),
            item_index=int(metadata["source_row_index"]),
        ),
    )


def external_output(audio: Audio) -> dict[str, Audio | SaveResult]:
    saved = replace(audio, virtual=True, byte_length=0)
    path = f"db/audio/{audio.audio_file_id}"
    return {
        "audio": saved,
        "save_result": SaveResult(
            Path(path),
            "external_audio_record",
            stable_id("save", path),
            audio.lineage_id,
        ),
    }


def _segment_dict(segment: AudioSegment) -> dict[str, Any]:
    return {
        "id": segment.segment_id or segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker or "",
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "confidence": segment.confidence,
        "type_": str(segment.metadata["type_"]),
        "metadata": segment.metadata,
        "alignment": segment.alignment,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
