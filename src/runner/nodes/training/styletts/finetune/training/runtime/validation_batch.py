import torch
import torch.nn.functional as F
from torch import nn

def styletts_zs_reconstruction_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    lengths: torch.Tensor | list[int],
    divisor: float = 1.0,
) -> torch.Tensor:
    """The length-normalized reconstruction objective used by StyleTTS-ZS."""
    length_total = torch.as_tensor(lengths, device=target.device).sum()
    valid_ratio = target.numel() / length_total
    return F.smooth_l1_loss(target, prediction) * valid_ratio / divisor


def acoustic_losses(
    stft_loss: nn.Module,
    reconstructed: torch.Tensor,
    waveform: torch.Tensor,
    ground_truth_samples: list[int],
    prediction_samples: list[int],
) -> torch.Tensor:
    mel_losses = []
    for index, (gt_samples, pred_samples) in enumerate(
        zip(
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
    return torch.stack(mel_losses).mean()


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
