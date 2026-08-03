import torch
import torch.nn.functional as F


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
