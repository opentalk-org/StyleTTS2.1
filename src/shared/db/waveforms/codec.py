import io
import struct
import wave

import numpy as np

from shared.db.waveforms.schemas import WaveformInput


FORMAT_VERSION = 1
DEFAULT_POINTS_PER_SECOND = 100
INT16_MAX = 32767


def encode_peaks(peaks: list[tuple[float, float]]) -> bytes:
    data = bytearray(len(peaks) * 4)
    for index, (minimum, maximum) in enumerate(peaks):
        struct.pack_into("<hh", data, index * 4, _quantize(minimum), _quantize(maximum))
    return bytes(data)


def decode_peaks(data: bytes) -> list[tuple[float, float]]:
    assert len(data) % 4 == 0, f"waveform byte length must align to int16 pairs: {len(data)}"
    return [
        (minimum / INT16_MAX, maximum / INT16_MAX)
        for minimum, maximum in struct.iter_unpack("<hh", data)
    ]


def waveform_from_wav(data: bytes, points_per_second: int = DEFAULT_POINTS_PER_SECOND) -> WaveformInput:
    with wave.open(io.BytesIO(data), "rb") as reader:
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames_per_point = max(1, sample_rate // points_per_second)
        peaks: list[tuple[float, float]] = []
        while True:
            chunk = reader.readframes(frames_per_point)
            if not chunk:
                break
            peaks.append(_chunk_peak(chunk, sample_width))
    return WaveformInput(sample_rate=sample_rate, points_per_second=points_per_second, peaks=peaks)


def waveform_from_audio_bytes(data: bytes, points_per_second: int = DEFAULT_POINTS_PER_SECOND) -> WaveformInput:
    samples, sample_rate = _decode_audio(data)
    step = max(1, sample_rate // points_per_second)
    frames = samples.shape[0]
    frame_min = samples.min(axis=1)
    frame_max = samples.max(axis=1)
    full = frames // step
    peaks: list[tuple[float, float]] = []
    if full:
        minima = frame_min[: full * step].reshape(full, step).min(axis=1)
        maxima = frame_max[: full * step].reshape(full, step).max(axis=1)
        peaks.extend(zip(minima.tolist(), maxima.tolist()))
    if frames % step:
        peaks.append((float(frame_min[full * step :].min()), float(frame_max[full * step :].max())))
    return WaveformInput(sample_rate=int(sample_rate), points_per_second=points_per_second, peaks=peaks)


def _decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile

        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except Exception:
        import librosa

        samples, sample_rate = librosa.load(io.BytesIO(data), sr=None, mono=False)
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(-1, 1), int(sample_rate)
        return array.T, int(sample_rate)


def downsample(peaks: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if max_points <= 0 or len(peaks) <= max_points:
        return peaks
    out: list[tuple[float, float]] = []
    for index in range(max_points):
        start = index * len(peaks) // max_points
        end = max(start + 1, (index + 1) * len(peaks) // max_points)
        bucket = peaks[start:end]
        out.append((min(item[0] for item in bucket), max(item[1] for item in bucket)))
    return out


def _chunk_peak(chunk: bytes, sample_width: int) -> tuple[float, float]:
    values = _samples(chunk, sample_width)
    return min(values), max(values)


def _samples(chunk: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(value - 128) / 128 for value in chunk]
    if sample_width == 2:
        return [value / 32768 for (value,) in struct.iter_unpack("<h", chunk)]
    if sample_width == 4:
        return [value / 2147483648 for (value,) in struct.iter_unpack("<i", chunk)]
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _quantize(value: float) -> int:
    bounded = max(-1.0, min(1.0, value))
    return int(round(bounded * INT16_MAX))
