from dataclasses import dataclass
from pathlib import Path

import torch
from monotonic_align import maximum_path
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import AlignerConfig


@dataclass(frozen=True)
class AlignerOutput:
    ctc_logits: Tensor
    s2s_logits: Tensor
    soft_alignment: Tensor
    hard_alignment: Tensor
    durations: Tensor


class PhonemeAligner(nn.Module):
    """Strict adapter for the pretrained StyleTTS ASRCNN output contract."""

    def __init__(
        self,
        backbone: nn.Module,
        config: AlignerConfig,
        backbone_vocabulary_size: int,
        frame_reduction: int,
    ) -> None:
        super().__init__()
        if backbone_vocabulary_size != config.vocabulary_size:
            raise ValueError("aligner backbone vocabulary does not match configuration")
        if config.blank_id >= config.vocabulary_size:
            raise ValueError("aligner blank id is outside the vocabulary")
        if frame_reduction <= 0:
            raise ValueError("aligner frame reduction must be positive")
        self.backbone = backbone
        self.config = config
        self.frame_reduction = frame_reduction

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload["model"]
        self.backbone.load_state_dict(state, strict=True)

    def forward(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        phonemes: Tensor,
        phoneme_mask: Tensor,
    ) -> AlignerOutput:
        if mel.ndim != 3 or frame_mask.shape != (mel.shape[0], 1, mel.shape[2]):
            raise ValueError("aligner requires [B,M,T] mel and [B,1,T] frame mask")
        if phonemes.ndim != 2 or phoneme_mask.shape != phonemes.shape:
            raise ValueError("aligner requires [B,P] phonemes and phoneme mask")
        if phonemes.shape[0] != mel.shape[0]:
            raise ValueError("aligner mel and phoneme batch sizes must match")
        frame_lengths = frame_mask.sum(dim=(1, 2))
        phoneme_lengths = phoneme_mask.sum(dim=1)
        if torch.any(phoneme_lengths > frame_lengths):
            raise ValueError("phoneme count must not exceed valid mel frames")
        reduced_frames = (
            mel.shape[2] + self.frame_reduction - 1
        ) // self.frame_reduction
        reduced_lengths = torch.div(
            frame_lengths + self.frame_reduction - 1,
            self.frame_reduction,
            rounding_mode="floor",
        )
        positions = torch.arange(reduced_frames, device=mel.device).unsqueeze(0)
        ctc_mask = positions < reduced_lengths.unsqueeze(1)
        ctc_logits, s2s_logits, raw_attention = self.backbone(
            mel,
            ~ctc_mask,
            phonemes,
        )
        if ctc_logits.shape != (
            mel.shape[0],
            reduced_frames,
            self.config.vocabulary_size,
        ):
            raise ValueError(
                "aligner CTC output does not match frame or vocabulary shape"
            )
        max_phonemes = phonemes.shape[1]
        if (
            s2s_logits.shape[0] != mel.shape[0]
            or s2s_logits.shape[2] != self.config.vocabulary_size
        ):
            raise ValueError(
                "aligner sequence output does not match batch or vocabulary"
            )
        if s2s_logits.shape[1] < max_phonemes:
            raise ValueError("aligner sequence output is shorter than phoneme input")
        if (
            raw_attention.shape[0] != mel.shape[0]
            or raw_attention.shape[1] < max_phonemes + 1
        ):
            raise ValueError(
                "aligner attention is missing the start row or phoneme rows"
            )
        soft_alignment = raw_attention[:, 1 : max_phonemes + 1]
        if soft_alignment.shape[2] != mel.shape[2]:
            soft_alignment = F.interpolate(
                soft_alignment,
                size=mel.shape[2],
                mode="linear",
                align_corners=False,
            )
        valid_matrix = phoneme_mask.unsqueeze(2) & frame_mask
        soft_alignment = soft_alignment * valid_matrix
        normalization = soft_alignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
        soft_alignment = soft_alignment / normalization * valid_matrix
        alignment_scores = torch.log(soft_alignment.clamp_min(1e-8))
        hard_alignment = maximum_path(
            alignment_scores,
            valid_matrix.to(dtype=alignment_scores.dtype),
        )
        hard_alignment = hard_alignment * valid_matrix
        durations = hard_alignment.sum(dim=2)
        return AlignerOutput(
            ctc_logits=ctc_logits,
            s2s_logits=s2s_logits[:, :max_phonemes],
            soft_alignment=soft_alignment,
            hard_alignment=hard_alignment,
            durations=durations,
        )
