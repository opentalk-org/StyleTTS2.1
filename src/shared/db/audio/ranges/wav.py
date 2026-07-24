from __future__ import annotations

import io
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class WavTimeRange:
    start: float
    end: float


@dataclass(frozen=True)
class WavClip:
    data: bytes
    sample_rate: int
    channels: int


def slice_wav_ranges(data: bytes, ranges: list[WavTimeRange]) -> list[WavClip]:
    with wave.open(io.BytesIO(data), "rb") as source:
        return _read_clips(source, ranges)


def _read_clips(
    source: wave.Wave_read,
    ranges: list[WavTimeRange],
) -> list[WavClip]:
    sample_rate = source.getframerate()
    channels = source.getnchannels()
    sample_width = source.getsampwidth()
    frame_count = source.getnframes()
    clips = []
    for item in ranges:
        start_frame = int(round(max(0.0, item.start) * sample_rate))
        end_frame = int(round(max(0.0, item.end) * sample_rate))
        end_frame = min(max(start_frame + 1, end_frame), frame_count)
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        clips.append(
            WavClip(
                _wav_bytes(frames, sample_rate, channels, sample_width),
                sample_rate,
                channels,
            )
        )
    return clips


def _wav_bytes(
    frames: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(frames)
    return output.getvalue()
