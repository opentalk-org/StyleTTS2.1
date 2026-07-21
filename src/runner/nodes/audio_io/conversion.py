from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ConvertedAudio:
    data: bytes
    sample_rate: int
    channels: int
    duration: float


def normalize_wav_bytes(data: bytes, sample_rate: int, channels: int) -> ConvertedAudio:
    samples, source_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError("audio payload contains no frames")
    converted = _convert_channels(np.asarray(samples, dtype=np.float32), channels)
    if int(source_rate) != sample_rate:
        converted = librosa.resample(
            converted.T,
            orig_sr=int(source_rate),
            target_sr=sample_rate,
            axis=-1,
        ).T
    output = BytesIO()
    sf.write(output, converted, sample_rate, format="WAV", subtype="PCM_16")
    return ConvertedAudio(
        data=output.getvalue(),
        sample_rate=sample_rate,
        channels=channels,
        duration=converted.shape[0] / float(sample_rate),
    )


def _convert_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    source_channels = samples.shape[1]
    if channels == 1:
        return samples.mean(axis=1, keepdims=True, dtype=np.float32)
    if source_channels >= channels:
        return samples[:, :channels]
    repeated = np.repeat(samples[:, -1:], channels - source_channels, axis=1)
    return np.concatenate((samples, repeated), axis=1)
