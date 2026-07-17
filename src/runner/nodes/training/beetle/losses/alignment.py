from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..models.modules.aligner import AlignerOutput


@dataclass(frozen=True)
class AlignmentLosses:
    s2s: Tensor
    mono: Tensor
    ctc: Tensor


def compute_alignment_losses(
    output: AlignerOutput,
    phonemes: Tensor,
    phoneme_mask: Tensor,
    frame_mask: Tensor,
    blank_id: int,
    frame_reduction: int,
) -> AlignmentLosses:
    if output.s2s_logits.shape[:2] != phonemes.shape:
        raise ValueError("sequence logits and phonemes must have equal token shapes")
    if phoneme_mask.shape != phonemes.shape:
        raise ValueError("phoneme mask must match phonemes")
    if frame_mask.shape != (
        output.soft_alignment.shape[0],
        1,
        output.soft_alignment.shape[2],
    ):
        raise ValueError("frame mask must match alignment frames")
    token_count = phoneme_mask.sum()
    if token_count == 0:
        raise ValueError("alignment loss requires a valid phoneme")
    sequence_loss = F.cross_entropy(
        output.s2s_logits.transpose(1, 2),
        phonemes,
        reduction="none",
    )
    s2s = (sequence_loss * phoneme_mask).sum() / token_count

    valid_matrix = phoneme_mask.unsqueeze(2) & frame_mask
    mono = (
        (output.soft_alignment - output.hard_alignment).abs() * valid_matrix
    ).sum() / valid_matrix.sum()
    mono = mono * 10

    frame_lengths = frame_mask.sum(dim=(1, 2))
    input_lengths = torch.div(
        frame_lengths + frame_reduction - 1,
        frame_reduction,
        rounding_mode="floor",
    )
    target_lengths = phoneme_mask.sum(dim=1)
    if torch.any(target_lengths > input_lengths):
        raise ValueError("CTC target length exceeds reduced input length")
    ctc = (
        F.ctc_loss(
            output.ctc_logits.log_softmax(dim=2).transpose(0, 1),
            phonemes.masked_select(phoneme_mask),
            input_lengths,
            target_lengths,
            blank=blank_id,
            reduction="sum",
            zero_infinity=False,
        )
        / token_count
    )
    return AlignmentLosses(s2s=s2s, mono=mono, ctc=ctc)
