from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as functional
import librosa
import numpy as np
import soundfile as sf
import torchaudio

from runner.nodes.models import Audio, CheckpointRef
from runner.nodes.text.runtime.symbols import TextCleaner
from runner.nodes.training.styletts.finetune.training.modules.asr.models import ASRCNN
from runner.nodes.synthesis.styletts_runtime.checkpoints import latest_weight


CALIBRATION_INTERCEPT = -2.3970031054401475
CALIBRATION_WEIGHTS = (0.9114070082754281, 4.265471256497811, -0.5883846323940787, -0.2612922762920061)
MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    n_fft=2048,
    win_length=1200,
    hop_length=300,
    n_mels=80,
)


@dataclass(frozen=True)
class TranscriptQualityMetrics:
    ctc_loss: float
    sequence_loss: float
    attention_entropy: float
    attention_peak: float
    combined: float
    score: float


def load_checkpoint_aligner(checkpoint: CheckpointRef, device: torch.device) -> tuple[ASRCNN, TextCleaner]:
    checkpoint_type = checkpoint.metadata["type"]
    if checkpoint_type != "styletts2":
        raise ValueError(f"transcript quality requires a styletts2 checkpoint, got {checkpoint_type}")
    state = torch.load(latest_weight(checkpoint.path), map_location="cpu", weights_only=False, mmap=True)
    aligner_state = state["net"]["text_aligner"]
    embedding = aligner_state["asr_s2s.embedding.weight"]
    model = ASRCNN(
        input_dim=80,
        hidden_dim=256,
        n_token=int(embedding.shape[0]),
        n_layers=6,
        token_embedding_dim=int(embedding.shape[1]),
    )
    model.load_state_dict(aligner_state, strict=True)
    model.asr_s2s.random_mask = 0.0
    model.eval().to(device)
    symbols = checkpoint.metadata["metadata"]["symbols"]
    return model, TextCleaner([str(symbol) for symbol in symbols])


def score_transcript_batch(
    model: ASRCNN,
    cleaner: TextCleaner,
    audios: list[Audio],
    device: torch.device,
) -> list[TranscriptQualityMetrics]:
    phonemes = [" ".join(segment.phon.strip() for segment in audio.segments if segment.phon.strip()) for audio in audios]
    if any(not phon for phon in phonemes):
        raise ValueError("transcript quality requires phonemized transcript segments")
    token_rows = [torch.tensor(cleaner(phon), dtype=torch.long) for phon in phonemes]
    if any(row.numel() == 0 for row in token_rows):
        raise ValueError("transcript quality phonemes contain no checkpoint symbols")
    mels = _load_mels(audios)
    mel_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    token_lengths = torch.tensor([row.numel() for row in token_rows], dtype=torch.long)
    encoded_lengths = mel_lengths // (2**model.n_down)
    if torch.any(token_lengths > encoded_lengths):
        raise ValueError("transcript has more phoneme tokens than aligner audio frames")
    mel_batch = _pad_mels(mels).to(device)
    token_batch = torch.nn.utils.rnn.pad_sequence(token_rows, batch_first=True).to(device)
    encoded_lengths = encoded_lengths.to(device)
    token_lengths = token_lengths.to(device)
    with torch.inference_mode():
        ctc_logits, sequence_logits, attention = model(
            mel_batch,
            src_key_padding_mask=model.length_to_mask(encoded_lengths),
            text_input=token_batch,
        )
        ctc = functional.ctc_loss(
            ctc_logits.log_softmax(2).transpose(0, 1),
            token_batch,
            encoded_lengths,
            token_lengths,
            blank=cleaner.symbol_index[" "],
            reduction="none",
            zero_infinity=True,
        ) / token_lengths
        sequence = _sequence_losses(sequence_logits, token_batch, token_lengths)
        entropy, peak = _attention_metrics(attention[:, 1:], token_lengths, encoded_lengths)
    return [
        _calibrated_metrics(float(ctc[index]), float(sequence[index]), float(entropy[index]), float(peak[index]))
        for index in range(len(audios))
    ]


def annotate_transcript_quality(audio: Audio, metrics: TranscriptQualityMetrics) -> Audio:
    values = {
        "transcript_error_score": metrics.score,
        "transcript_quality_metrics": {
            "ctc_loss": metrics.ctc_loss,
            "sequence_loss": metrics.sequence_loss,
            "attention_entropy": metrics.attention_entropy,
            "attention_peak": metrics.attention_peak,
            "combined": metrics.combined,
        },
    }
    annotations = audio.annotations.model_copy(update={"metadata": {**audio.metadata, **values}})
    return replace(audio, annotations=annotations)


def _load_mels(audios: list[Audio]) -> list[torch.Tensor]:
    assert all(audio.data is not None for audio in audios), "transcript quality requires audio bytes"
    with ThreadPoolExecutor(max_workers=min(8, len(audios)), thread_name_prefix="transcript-quality") as pool:
        return list(pool.map(_load_mel, (audio.data for audio in audios)))


def _load_mel(data: bytes) -> torch.Tensor:
    wave, sample_rate = sf.read(BytesIO(data), always_2d=True, dtype="float32")
    mono = wave[:, 0]
    if sample_rate != 24_000:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=24_000)
    padded = np.concatenate((np.zeros(5_000), mono, np.zeros(5_000)))
    mel = MEL_TRANSFORM(torch.from_numpy(padded).float())
    normalized = (torch.log(1e-5 + mel) + 4.0) / 4.0
    return normalized[:, : normalized.shape[1] - normalized.shape[1] % 2]


def _pad_mels(mels: list[torch.Tensor]) -> torch.Tensor:
    output = torch.zeros((len(mels), 80, max(mel.shape[1] for mel in mels)), dtype=torch.float32)
    for index, mel in enumerate(mels):
        output[index, :, : mel.shape[1]] = mel
    return output


def _sequence_losses(logits: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    losses = functional.cross_entropy(logits[:, :-1].transpose(1, 2), tokens, reduction="none")
    mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
    return (losses * mask).sum(1) / lengths


def _attention_metrics(
    attention: torch.Tensor,
    token_lengths: torch.Tensor,
    encoded_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_mask = torch.arange(attention.shape[1], device=attention.device).unsqueeze(0) < token_lengths.unsqueeze(1)
    frame_mask = torch.arange(attention.shape[2], device=attention.device).unsqueeze(0) < encoded_lengths.unsqueeze(1)
    valid = token_mask.unsqueeze(2) & frame_mask.unsqueeze(1)
    aligned = attention.masked_fill(~valid, 0.0)
    entropy = -(aligned.clamp_min(1e-9) * aligned.clamp_min(1e-9).log()).sum(2)
    return (entropy * token_mask).sum(1) / token_lengths, aligned.max(2).values.sum(1) / token_lengths


def _calibrated_metrics(ctc: float, sequence: float, entropy: float, peak: float) -> TranscriptQualityMetrics:
    evidence = (ctc, sequence, entropy, 1.0 - peak)
    combined = CALIBRATION_INTERCEPT + sum(
        weight * value for weight, value in zip(CALIBRATION_WEIGHTS, evidence, strict=True)
    )
    score = 1.0 / (1.0 + math.exp(-combined))
    return TranscriptQualityMetrics(ctc, sequence, entropy, peak, combined, score)
