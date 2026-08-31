from __future__ import annotations

from pathlib import Path

import torch

from runner.nodes.asr.transcript_quality import load_checkpoint_aligner, score_transcript_batch
from runner.nodes.asr.audio import wav_info
from runner.nodes.models import Audio, AudioSegment, CheckpointRef, stable_id
from shared.audio_annotations import AudioAnnotations

from .assets import ResolvedAssets
from .config import AssetConfig
from .data import BackendAudioRow


class FrozenAlignerLoss:
    def __init__(self, config: AssetConfig, assets: ResolvedAssets, device: torch.device) -> None:
        checkpoint = CheckpointRef(
            config.aligner_checkpoint_id,
            "frozen-styletts-aligner",
            assets.aligner_root,
            stable_id("checkpoint", config.aligner_checkpoint_id),
            stable_id("checkpoint", config.aligner_checkpoint_id),
            assets.aligner_metadata,
        )
        self.model, self.cleaner = load_checkpoint_aligner(checkpoint, device)
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, rows: list[BackendAudioRow], phonemes: list[str]) -> torch.Tensor:
        audios = [_audio(row, phoneme) for row, phoneme in zip(rows, phonemes, strict=True)]
        try:
            metrics = score_transcript_batch(self.model, self.cleaner, audios, self.device)
            losses = [metric.combined for metric in metrics]
        except ValueError as error:
            if str(error) != "transcript has more phoneme tokens than aligner audio frames":
                raise
            losses = [self._score_or_overlength_loss(audio, phoneme) for audio, phoneme in zip(audios, phonemes, strict=True)]
        return torch.tensor(losses, device=self.device)

    def _score_or_overlength_loss(self, audio: Audio, phoneme: str) -> float:
        try:
            return score_transcript_batch(self.model, self.cleaner, [audio], self.device)[0].combined
        except ValueError as error:
            if str(error) != "transcript has more phoneme tokens than aligner audio frames":
                raise
            return float(len(self.cleaner(phoneme)))


def _audio(row: BackendAudioRow, phoneme: str) -> Audio:
    info = wav_info(row.wav_bytes)
    duration = info["frame_count"] / info["sample_rate"]
    annotations = AudioAnnotations(metadata={"source": "bert_g2p_asr_ppo"})
    segment = AudioSegment(
        row.audio_id,
        str(row.audio_id),
        0.0,
        duration,
        info["sample_rate"],
        info["channels"],
        row.text,
        phoneme,
        stable_id("segment", row.audio_id),
        stable_id("segment_lineage", row.audio_id),
        annotations=annotations,
    )
    return Audio(
        row.audio_id,
        str(row.audio_id),
        row.wav_bytes,
        info["sample_rate"],
        info["channels"],
        0.0,
        duration,
        annotations,
        stable_id("audio", row.audio_id),
        stable_id("audio_lineage", row.audio_id),
        row.language,
        byte_length=len(row.wav_bytes),
        segments=[segment],
    )
