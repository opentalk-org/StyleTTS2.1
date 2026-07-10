from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

import librosa
import numpy as np
import soundfile as sf
import torch


MOS_SAMPLE_RATE = 16_000


class MosFeatureExtractor(Protocol):
    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        padding: bool,
        return_attention_mask: bool,
        return_tensors: str,
    ): ...


@dataclass(frozen=True)
class MosInputs:
    input_values: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> MosInputs:
        return MosInputs(
            input_values=self.input_values.to(device),
            attention_mask=self.attention_mask.to(device),
        )


def decode_audio_bytes(data: bytes, target_sample_rate: int = MOS_SAMPLE_RATE) -> np.ndarray:
    audio, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    if not audio.size:
        raise ValueError("MOS audio payload is empty")
    mono = audio.mean(axis=1, dtype=np.float32)
    if sample_rate != target_sample_rate:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sample_rate)
    return np.asarray(mono, dtype=np.float32)


def prepare_audio_batch(feature_extractor: MosFeatureExtractor, audio_bytes: list[bytes]) -> MosInputs:
    if not audio_bytes:
        raise ValueError("MOS audio batch is empty")
    waveforms = [decode_audio_bytes(data) for data in audio_bytes]
    encoded = feature_extractor(
        waveforms,
        sampling_rate=MOS_SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return MosInputs(
        input_values=encoded.input_values,
        attention_mask=encoded.attention_mask,
    )
