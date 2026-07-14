from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SilenceInterval:
    start: float
    end: float


def detect_silence_intervals(
    data: bytes,
    audio_start: float,
    threshold: float,
    window_size_ms: int,
    max_gap_ms: int,
) -> list[SilenceInterval]:
    samples, sample_rate = sf.read(BytesIO(data), always_2d=True, dtype="float32")
    if len(samples) == 0:
        return []
    mono = np.mean(samples, axis=1, dtype=np.float64)
    window_samples = max(1, int(round(int(sample_rate) * window_size_ms / 1000.0)))
    max_gap_samples = int(round(int(sample_rate) * max_gap_ms / 1000.0))
    silent_windows = _silent_windows(mono, window_samples, threshold)
    return _merged_intervals(silent_windows, len(mono), int(sample_rate), audio_start, max_gap_samples)


def _silent_windows(samples: np.ndarray, window_samples: int, threshold: float) -> list[tuple[int, int]]:
    windows = []
    for start in range(0, len(samples), window_samples):
        end = min(len(samples), start + window_samples)
        rms = float(np.sqrt(np.mean(np.square(samples[start:end]))))
        if rms <= threshold:
            windows.append((start, end))
    return windows


def _merged_intervals(
    windows: list[tuple[int, int]],
    sample_count: int,
    sample_rate: int,
    audio_start: float,
    max_gap_samples: int,
) -> list[SilenceInterval]:
    if not windows:
        return []
    merged: list[tuple[int, int]] = []
    current_start, current_end = windows[0]
    for start, end in windows[1:]:
        if start - current_end <= max_gap_samples:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, min(current_end, sample_count)))
    return [
        SilenceInterval(audio_start + start / sample_rate, audio_start + end / sample_rate)
        for start, end in merged
    ]
