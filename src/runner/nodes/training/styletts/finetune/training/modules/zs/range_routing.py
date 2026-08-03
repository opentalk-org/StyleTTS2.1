from typing import Sequence

import torch
from torch import Tensor


def route_candidates(candidates: Tensor, assignments: Tensor) -> Tensor:
    indices = assignments[:, None, None, :].expand(-1, 1, candidates.size(2), -1)
    return candidates.gather(1, indices).squeeze(1)


def resolve_ranges(
    phoneme_count: int,
    ranges: Sequence[tuple[int, int, int]],
    default_reference: int = 0,
) -> Tensor:
    assignments = torch.full((phoneme_count,), default_reference, dtype=torch.long)
    claimed = torch.zeros(phoneme_count, dtype=torch.bool)
    for start, end, reference in ranges:
        if not 0 <= start < end <= phoneme_count:
            raise ValueError(f"invalid phoneme range [{start}, {end}) for length {phoneme_count}")
        if claimed[start:end].any():
            raise ValueError(f"overlapping phoneme range [{start}, {end})")
        assignments[start:end] = reference
        claimed[start:end] = True
    return assignments


def expand_assignments(assignments: Tensor, alignment: Tensor, scale_factor: int = 1) -> Tensor:
    if assignments.shape != alignment.shape[:2]:
        raise ValueError("assignments and alignment phoneme dimensions must match")
    frames = torch.einsum("bn,bnt->bt", assignments.to(alignment.dtype), alignment)
    frame_assignments = frames.round().long()
    if scale_factor > 1:
        frame_assignments = frame_assignments.repeat_interleave(scale_factor, dim=-1)
    return frame_assignments


def route_voice_candidates(
    candidates: Tensor,
    assignments: Tensor,
    alignment: Tensor,
    smoothing_width: int = 4,
) -> Tensor:
    indices = assignments[:, :, None].expand(-1, -1, candidates.size(-1))
    phoneme_track = candidates.gather(1, indices).transpose(1, 2)
    frame_track = phoneme_track @ alignment
    frame_assignments = expand_assignments(assignments, alignment)
    return smooth_feature_boundaries(frame_track, frame_assignments, smoothing_width)


def smooth_feature_boundaries(track: Tensor, assignments: Tensor, width: int = 4) -> Tensor:
    smoothed = track.clone()
    for batch_index in range(track.size(0)):
        boundaries = torch.nonzero(assignments[batch_index, 1:] != assignments[batch_index, :-1]).flatten() + 1
        for boundary_value in boundaries:
            boundary = int(boundary_value.item())
            left_start, right_end = _neighbor_bounds(assignments[batch_index], boundary)
            radius = min(width, (boundary - left_start) // 2, (right_end - boundary + 1) // 2)
            if radius == 0:
                continue
            weights = 0.5 - 0.5 * torch.cos(
                torch.linspace(0, torch.pi, radius * 2, device=track.device, dtype=track.dtype)
            )
            left = track[batch_index, :, boundary - radius][:, None]
            right = track[batch_index, :, boundary][:, None]
            smoothed[batch_index, :, boundary - radius : boundary + radius] = left * (1 - weights) + right * weights
    return smoothed


def _neighbor_bounds(assignments: Tensor, boundary: int) -> tuple[int, int]:
    left_start = boundary - 1
    while left_start > 0 and assignments[left_start - 1] == assignments[boundary - 1]:
        left_start -= 1
    right_end = boundary
    while right_end < assignments.size(0) - 1 and assignments[right_end + 1] == assignments[boundary]:
        right_end += 1
    return left_start, right_end


def smooth_prosody_boundaries(
    f0: Tensor,
    energy: Tensor,
    assignments: Tensor,
    width: int = 4,
) -> tuple[Tensor, Tensor]:
    smoothed_energy = smooth_feature_boundaries(energy[:, None], assignments, width).squeeze(1)
    smoothed_f0 = smooth_feature_boundaries(f0[:, None], assignments, width).squeeze(1)
    incompatible = (assignments[:, 1:] != assignments[:, :-1]) & ~(
        (f0[:, 1:] > 0) & (f0[:, :-1] > 0)
    )
    for batch_index, values in enumerate(incompatible):
        for boundary in (torch.nonzero(values).flatten() + 1).tolist():
            smoothed_f0[batch_index, boundary - 1 : boundary + 1] = f0[
                batch_index, boundary - 1 : boundary + 1
            ]
    return smoothed_f0, smoothed_energy
