from __future__ import annotations

from io import BytesIO
from typing import Any

from pydantic import Field

from runflow.core.settings import StrictSettings
from runner.nodes.asr.audio import extract_wav_range, wav_info
from runner.nodes.models import Audio, stable_id


class VadSettings(StrictSettings):
    min_segment_sec: float = Field(default=1.0, ge=0.1, le=600.0)
    max_segment_sec: float = Field(default=12.0, ge=1.0, le=600.0)
    padding_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    max_silence_gap_ms: int = Field(default=400, ge=50, le=30000)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    hop_length: int = Field(default=512, ge=64, le=4096)


def vad_segments_batch(
    audios: list[Audio],
    settings: VadSettings,
) -> list[list[Audio]]:
    dependencies = _audio_dependencies()
    return [vad_segments(audio, settings, dependencies) for audio in audios]


def vad_segments(
    audio: Audio,
    settings: VadSettings,
    dependencies: tuple[Any, Any] | None = None,
) -> list[Audio]:
    librosa, np = dependencies if dependencies is not None else _audio_dependencies()
    y, sr = librosa.load(BytesIO(audio.data), sr=None, mono=True)
    samples = np.asarray(y, dtype=np.float32)
    if samples.size == 0:
        return []
    intervals = librosa.effects.split(
        samples,
        top_db=abs(float(settings.silence_threshold_db)),
        frame_length=2048,
        hop_length=settings.hop_length,
    )
    spans = _split_long_spans(
        _merge_spans(
            [
                (
                    max(0.0, float(start) / float(sr) - settings.padding_sec),
                    min(audio.duration, float(end) / float(sr) + settings.padding_sec),
                )
                for start, end in intervals
            ],
            settings.max_silence_gap_ms / 1000.0,
        ),
        settings.max_segment_sec,
    )
    valid = [(start, end) for start, end in spans if end - start >= settings.min_segment_sec]
    info = wav_info(audio.data)
    return [_segment_audio(audio, start, end, info, index) for index, (start, end) in enumerate(valid)]


def _segment_audio(
    audio: Audio,
    start: float,
    end: float,
    info: dict[str, int],
    index: int,
) -> Audio:
    data = extract_wav_range(audio.data, start, end, info)
    absolute_start = audio.start + start
    absolute_end = audio.start + end
    segment_id = stable_id("audio", audio.audio_file_id, audio.id, "vad", index, absolute_start, absolute_end)
    return Audio(
        audio.audio_file_id,
        audio.name,
        data,
        int(info["sample_rate"]),
        int(info["channels"]),
        absolute_start,
        absolute_end,
        audio.annotations.model_copy(update={"metadata": {
            **audio.metadata,
            "vad": {
                "source_audio_id": audio.id,
                "segment_index": index,
                "start": absolute_start,
                "end": absolute_end,
            },
        }}),
        segment_id,
        stable_id("lineage", audio.lineage_id, segment_id),
    )


def _merge_spans(
    spans: list[tuple[float, float]],
    max_gap_seconds: float,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= max_gap_seconds:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_long_spans(
    spans: list[tuple[float, float]],
    max_segment_sec: float,
) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    for start, end in spans:
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_segment_sec)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
    return chunks


def _audio_dependencies() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "VadDetect requires optional dependencies 'librosa' and 'numpy'"
        ) from error
    return librosa, np
