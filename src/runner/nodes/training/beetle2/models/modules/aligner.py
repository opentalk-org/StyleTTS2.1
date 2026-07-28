from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import torch
from monotonic_align import maximum_path
from torch import Tensor, nn

from ...config.architecture import AlignerConfig


logger = logging.getLogger(__name__)


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
        token_count: int,
        frame_reduction: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.token_count = token_count
        self.frame_reduction = frame_reduction

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        source_symbols: Sequence[str],
        target_symbols: Sequence[str],
    ) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["model"]
        target = self.backbone.state_dict()
        vocabulary_weights = (
            "ctc_linear.2.linear_layer.weight",
            "ctc_linear.2.linear_layer.bias",
            "asr_s2s.embedding.weight",
            "asr_s2s.project_to_n_symbols.weight",
            "asr_s2s.project_to_n_symbols.bias",
        )
        source_ids = {
            symbol: index for index, symbol in enumerate(source_symbols)
        }
        copied = 0
        for name in vocabulary_weights:
            resized = target[name]
            source = state[name]
            for target_id, symbol in enumerate(target_symbols):
                source_id = source_ids.get(symbol)
                if source_id is not None:
                    resized[target_id].copy_(source[source_id])
                    if name == vocabulary_weights[0]:
                        copied += 1
            state[name] = resized
        special_rows = {0: 0, 1: 1, 2: 3}
        for name in vocabulary_weights:
            for target_id, source_id in special_rows.items():
                state[name][target_id].copy_(payload["model"][name][source_id])
        self.backbone.load_state_dict(state, strict=True)
        logger.info(
            "resized aligner vocabulary from %d to %d tokens; retained %d symbol rows",
            len(source_symbols),
            len(target_symbols),
            copied,
        )

    def forward(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        phonemes: Tensor,
        phoneme_mask: Tensor,
    ) -> AlignerOutput:
        frame_lengths = frame_mask.sum(dim=(1, 2))
        phoneme_lengths = phoneme_mask.sum(dim=1)
        torch._assert_async(
            torch.all(phoneme_lengths <= frame_lengths),
            "phoneme count must not exceed valid mel frames",
        )
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
        alignment_frame_mask = ctc_mask.unsqueeze(1)
        ctc_logits, s2s_logits, raw_attention = self.backbone(
            mel,
            ~ctc_mask,
            phonemes,
        )
        max_phonemes = phonemes.shape[1]
        soft_alignment = raw_attention[:, 1 : max_phonemes + 1]
        valid_matrix = phoneme_mask.unsqueeze(2) & alignment_frame_mask
        soft_alignment = soft_alignment * valid_matrix
        normalization = soft_alignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
        soft_alignment = soft_alignment / normalization * valid_matrix
        alignment_scores = torch.log(soft_alignment.clamp_min(1e-8))
        hard_alignment = maximum_path(
            alignment_scores.contiguous(),
            valid_matrix.to(dtype=alignment_scores.dtype).contiguous(),
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
