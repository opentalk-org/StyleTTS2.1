from dataclasses import dataclass

import torch
from torch import Tensor

from .models.modules.audio import AudioPosterior


@dataclass(frozen=True)
class WindowRanges:
    target_starts: Tensor
    target_requested_lengths: Tensor
    target_source_lengths: Tensor
    pre_starts: Tensor
    pre_requested_lengths: Tensor
    pre_source_lengths: Tensor
    post_starts: Tensor
    post_requested_lengths: Tensor
    post_source_lengths: Tensor
    full_selected: Tensor


@dataclass(frozen=True)
class AlignedWindow:
    posterior: AudioPosterior
    aligned_phonemes: Tensor
    pre_audio: Tensor
    pre_audio_mask: Tensor
    post_audio: Tensor
    post_audio_mask: Tensor
    pre_phoneme_mask: Tensor
    post_phoneme_mask: Tensor
    ranges: WindowRanges


def apply_window_ranges(
    posterior: AudioPosterior,
    aligned_phonemes: Tensor,
    hard_alignment: Tensor,
    ranges: WindowRanges,
) -> AlignedWindow:
    mean, _ = slice_windows(
        posterior.mean,
        ranges.target_starts,
        ranges.target_requested_lengths,
        ranges.target_source_lengths,
        left_pad=False,
    )
    log_scale, _ = slice_windows(
        posterior.log_scale,
        ranges.target_starts,
        ranges.target_requested_lengths,
        ranges.target_source_lengths,
        left_pad=False,
    )
    latent, target_selection = slice_windows(
        posterior.latent,
        ranges.target_starts,
        ranges.target_requested_lengths,
        ranges.target_source_lengths,
        left_pad=False,
    )
    target_mask_values, _ = slice_windows(
        posterior.mask,
        ranges.target_starts,
        ranges.target_requested_lengths,
        ranges.target_source_lengths,
        left_pad=False,
    )
    target_mask = target_selection & target_mask_values.to(dtype=torch.bool)
    aligned, _ = slice_windows(
        aligned_phonemes,
        ranges.target_starts,
        ranges.target_requested_lengths,
        ranges.target_source_lengths,
        left_pad=False,
    )
    pre_audio, pre_mask = slice_windows(
        posterior.latent,
        ranges.pre_starts,
        ranges.pre_requested_lengths,
        ranges.pre_source_lengths,
        left_pad=True,
    )
    post_audio, post_mask = slice_windows(
        posterior.latent,
        ranges.post_starts,
        ranges.post_requested_lengths,
        ranges.post_source_lengths,
        left_pad=False,
    )
    pre_frames = range_mask(
        posterior.mask.shape[2],
        ranges.pre_starts,
        ranges.pre_source_lengths,
    )
    post_frames = range_mask(
        posterior.mask.shape[2],
        ranges.post_starts,
        ranges.post_source_lengths,
    )
    return AlignedWindow(
        AudioPosterior(mean, log_scale, latent, target_mask),
        aligned,
        pre_audio,
        pre_mask,
        post_audio,
        post_mask,
        phoneme_mask_for_frames(hard_alignment, pre_frames),
        phoneme_mask_for_frames(hard_alignment, post_frames),
        ranges,
    )


def range_mask(width: int, starts: Tensor, lengths: Tensor) -> Tensor:
    positions = torch.arange(width, device=starts.device).unsqueeze(0)
    mask = (positions >= starts.unsqueeze(1)) & (
        positions < (starts + lengths).unsqueeze(1)
    )
    return mask.unsqueeze(1)


def safe_context_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    available = mask.any(dim=2, keepdim=True)
    safe = mask.clone()
    safe[:, :, 0] |= ~available[:, :, 0]
    return safe, available


def seconds_to_latent_frames(
    seconds: float,
    sample_rate: int,
    hop_length: int,
    downsample_rate: int,
) -> int:
    samples = seconds * sample_rate
    return max(1, round(samples / (hop_length * downsample_rate)))


def slice_windows(
    values: Tensor,
    starts: Tensor,
    requested_lengths: Tensor,
    source_lengths: Tensor,
    left_pad: bool,
) -> tuple[Tensor, Tensor]:
    batch_size = values.shape[0]
    vector_shape = (batch_size,)
    width = int(requested_lengths.max().item())
    positions = torch.arange(width, device=values.device).unsqueeze(0)
    padding = requested_lengths - source_lengths if left_pad else torch.zeros_like(starts)
    relative = positions - padding.unsqueeze(1)
    valid = (
        (positions < requested_lengths.unsqueeze(1))
        & (relative >= 0)
        & (relative < source_lengths.unsqueeze(1))
    )
    indices = starts.unsqueeze(1) + relative.clamp_min(0)
    indices = indices.clamp_max(values.shape[2] - 1)
    gathered = torch.gather(
        values,
        2,
        indices.unsqueeze(1).expand(-1, values.shape[1], -1),
    )
    mask = valid.unsqueeze(1)
    return gathered * mask.to(dtype=values.dtype), mask


def phoneme_mask_for_frames(hard_alignment: Tensor, frame_mask: Tensor) -> Tensor:
    expected = (hard_alignment.shape[0], 1, hard_alignment.shape[2])
    selected = hard_alignment.to(dtype=torch.bool) & frame_mask
    return selected.any(dim=2, keepdim=False).unsqueeze(1)


def sample_window_ranges(
    valid_lengths: Tensor,
    minimum_frames: int,
    maximum_frames: int | None,
    context_minimum_frames: int,
    context_maximum_frames: int,
    full_audio_ratio: float,
    generator: torch.Generator,
    force_full: bool,
) -> WindowRanges:
    target_upper = (
        valid_lengths
        if maximum_frames is None
        else valid_lengths.clamp_max(maximum_frames)
    )
    padded_upper = target_upper.clamp_min(minimum_frames)
    sampled_lengths = _uniform_lengths(
        minimum_frames,
        padded_upper,
        generator,
    )
    full_selected = torch.rand(
        valid_lengths.shape,
        device=valid_lengths.device,
        generator=generator,
    ) < full_audio_ratio
    if force_full:
        full_selected = torch.ones_like(full_selected)
    requested = torch.where(full_selected, padded_upper, sampled_lengths)
    source = torch.minimum(requested, valid_lengths)
    available_starts = valid_lengths - source
    starts = _uniform_lengths(0, available_starts, generator)
    if force_full:
        starts = torch.zeros_like(starts)
    pre_available = starts
    post_starts = starts + source
    post_available = valid_lengths - post_starts
    pre_requested, pre_source = _context_lengths(
        pre_available,
        context_minimum_frames,
        context_maximum_frames,
        generator,
    )
    post_requested, post_source = _context_lengths(
        post_available,
        context_minimum_frames,
        context_maximum_frames,
        generator,
    )
    if force_full:
        pre_source = torch.zeros_like(pre_source)
        post_source = torch.zeros_like(post_source)
    return WindowRanges(
        target_starts=starts,
        target_requested_lengths=requested,
        target_source_lengths=source,
        pre_starts=starts - pre_source,
        pre_requested_lengths=pre_requested,
        pre_source_lengths=pre_source,
        post_starts=post_starts,
        post_requested_lengths=post_requested,
        post_source_lengths=post_source,
        full_selected=full_selected,
    )


def _context_lengths(
    available: Tensor,
    minimum: int,
    maximum: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    upper = available.clamp_min(minimum).clamp_max(maximum)
    requested = _uniform_lengths(minimum, upper, generator)
    return requested, torch.minimum(requested, available)


def _uniform_lengths(
    minimum: int,
    maximum: Tensor,
    generator: torch.Generator,
) -> Tensor:
    values = torch.rand(
        maximum.shape,
        device=maximum.device,
        generator=generator,
    )
    span = maximum - minimum + 1
    return minimum + torch.floor(values * span).to(dtype=maximum.dtype)
