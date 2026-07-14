from __future__ import annotations

from io import BytesIO
from typing import Protocol

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from runner.nodes.models import Audio
from runner.nodes.speaker_clustering.embedding_rows import PreparedSpeakerBatch


ECAPA_SAMPLE_RATE = 16_000
ECAPA_EMBEDDING_DIMENSION = 192


class ECAPAEncoder(Protocol):
    def encode_batch(self, waveforms: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor: ...


def prepare_ecapa_batch(audios: list[Audio]) -> PreparedSpeakerBatch:
    """Decode, normalize, duration-sort, and pad one bounded inference batch."""
    if not audios:
        raise ValueError("ECAPA batch requires at least one audio item")

    waveforms = [_prepare_waveform(audio) for audio in audios]
    order = sorted(range(len(waveforms)), key=lambda index: (-waveforms[index].numel(), index))
    sorted_waveforms = [waveforms[index] for index in order]
    sample_counts = torch.tensor([waveform.numel() for waveform in sorted_waveforms], dtype=torch.float32)
    padded = pad_sequence(sorted_waveforms, batch_first=True)
    return PreparedSpeakerBatch(
        waveforms=padded,
        relative_lengths=sample_counts / float(padded.shape[1]),
        original_indices=torch.tensor(order, dtype=torch.int64),
    )


def _prepare_waveform(audio: Audio) -> torch.Tensor:
    if audio.data is None:
        raise ValueError(f"ECAPA audio bytes are required: {audio.id}")
    waveform, sample_rate = torchaudio.load(BytesIO(audio.data))
    mono = waveform.to(dtype=torch.float32).mean(dim=0)
    if sample_rate != ECAPA_SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, ECAPA_SAMPLE_RATE)
    if mono.numel() == 0:
        raise ValueError(f"ECAPA audio is empty: {audio.id}")
    if not torch.isfinite(mono).all():
        raise ValueError(f"ECAPA audio contains non-finite samples: {audio.id}")
    return mono.contiguous()


class ECAPARuntime:
    def __init__(self, encoder: ECAPAEncoder) -> None:
        self._encoder = encoder

    def embed(self, batch: PreparedSpeakerBatch) -> np.ndarray:
        """Encode one prepared batch and restore unit vectors to caller order."""
        with torch.inference_mode():
            encoded = self._encoder.encode_batch(batch.waveforms, batch.relative_lengths)
        vectors = encoded.detach().to(device="cpu", dtype=torch.float32)
        if vectors.ndim == 3 and vectors.shape[1] == 1:
            vectors = vectors[:, 0, :]
        expected_shape = (batch.waveforms.shape[0], ECAPA_EMBEDDING_DIMENSION)
        if tuple(vectors.shape) != expected_shape:
            raise ValueError(f"ECAPA encoder returned shape {tuple(vectors.shape)}, expected {expected_shape}")
        if not torch.isfinite(vectors).all():
            raise ValueError("ECAPA encoder returned non-finite embeddings")

        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        if torch.any(norms == 0):
            raise ValueError("ECAPA encoder returned a zero-norm embedding")
        normalized = (vectors / norms).numpy()
        restored = np.empty_like(normalized, dtype=np.float32)
        restored[batch.original_indices.numpy()] = normalized
        return restored
