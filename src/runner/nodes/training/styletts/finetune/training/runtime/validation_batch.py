from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ValidationBatch:
    aligned_text: torch.Tensor
    starts: list[int]
    frames: int
    mel: torch.Tensor
    waveform: torch.Tensor
    lengths: list[int]
    sample_lengths: list[int]


def acoustic_losses(
    stft_loss: nn.Module,
    reconstructed: torch.Tensor,
    waveform: torch.Tensor,
    predicted_f0: torch.Tensor,
    target_f0: torch.Tensor,
    lengths: list[int],
    ground_truth_samples: list[int],
    prediction_samples: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    mel_losses = []
    f0_losses = []
    for index, (length, gt_samples, pred_samples) in enumerate(
        zip(
            lengths,
            ground_truth_samples,
            prediction_samples,
            strict=True,
        )
    ):
        samples = max(gt_samples, pred_samples)
        prediction = F.pad(
            reconstructed[index, 0, :pred_samples],
            (0, samples - pred_samples),
        ).unsqueeze(0)
        target = F.pad(
            waveform[index, 0, :gt_samples],
            (0, samples - gt_samples),
        ).unsqueeze(0)
        mel_losses.append(
            stft_loss(prediction, target).mean()
        )
        f0_losses.append(
            F.l1_loss(
                target_f0[index, :length],
                predicted_f0[index, :length],
            )
            / 10
        )
    return torch.stack(mel_losses).mean(), torch.stack(f0_losses).mean()


def predicted_alignment(
    duration_predictions: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    durations = torch.round(
        torch.sigmoid(duration_predictions).sum(dim=-1)
    ).clamp_min(1)
    token_positions = torch.arange(
        durations.size(1),
        device=durations.device,
    )
    input_lengths = input_lengths.to(durations.device)
    durations = durations * (
        token_positions.unsqueeze(0) < input_lengths.unsqueeze(1)
    )
    ends = durations.cumsum(dim=1)
    starts = ends - durations
    lengths = [int(value.item()) for value in ends[:, -1]]
    frames = torch.arange(max(lengths), device=durations.device)
    alignment = (
        (frames[None, None, :] >= starts[:, :, None])
        & (frames[None, None, :] < ends[:, :, None])
    )
    return alignment.to(duration_predictions.dtype), lengths


def resize_prosody(
    values: torch.Tensor,
    source_lengths: list[int],
    target_lengths: list[int],
) -> torch.Tensor:
    resized = values.new_zeros((values.size(0), max(target_lengths)))
    for index, (source, target) in enumerate(
        zip(source_lengths, target_lengths, strict=True)
    ):
        resized[index, :target] = F.interpolate(
            values[index, :source].reshape(1, 1, source),
            size=target,
            mode="linear",
            align_corners=False,
        ).reshape(target)
    return resized
