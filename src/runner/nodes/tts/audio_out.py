from __future__ import annotations

import io
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import soundfile as sf

from runner.nodes.models import Audio, stable_id


def wav_bytes_from_samples(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono waveform in [-1, 1] to 16-bit PCM WAV bytes."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(mono, -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def samples_from_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a float32 mono waveform and its sample rate."""
    array, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = array.mean(axis=1).astype(np.float32)
    return mono, int(sample_rate)


def audio_from_samples(
    *,
    node_type: str,
    request_id: str,
    output_name: str,
    samples: np.ndarray,
    sample_rate: int,
    metadata: dict[str, object],
) -> Audio:
    """Build an Audio artifact (with inline WAV bytes) from a synthesized waveform."""
    wav_bytes = wav_bytes_from_samples(samples, sample_rate)
    duration = len(np.asarray(samples).reshape(-1)) / float(sample_rate)
    audio_id = stable_id("audio", request_id)
    return Audio(
        audio_file_id=uuid5(NAMESPACE_URL, request_id),
        name=output_name,
        data=wav_bytes,
        sample_rate=sample_rate,
        channels=1,
        start=0.0,
        end=duration,
        confidence=1.0,
        id=audio_id,
        lineage_id=audio_id,
        metadata={"node_type": node_type, "request_id": request_id, "byte_length": len(wav_bytes), **metadata},
    )
