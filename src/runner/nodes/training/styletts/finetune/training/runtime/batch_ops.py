import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import TrainingBatch


def prosody_inputs(
    position_embedding,
    alignment: Tensor,
    f0: Tensor,
    energy: Tensor,
) -> Tensor:
    positions = torch.arange(alignment.size(1), device=alignment.device)
    positions = positions.unsqueeze(0).expand(alignment.size(0), -1)
    position_features = position_embedding(positions).transpose(1, 2) @ alignment
    half_f0 = F.avg_pool1d(f0.unsqueeze(1), 2)
    half_energy = F.avg_pool1d(energy.unsqueeze(1), 2)
    return torch.cat((position_features, half_energy, half_f0), dim=1)


def sample_alpha_flow_features(
    batch: TrainingBatch,
    text_encoding: Tensor,
    soft_alignment: Tensor,
    monotonic_alignment: Tensor,
    f0: Tensor,
    energy: Tensor,
) -> Tensor:
    alignment = (
        soft_alignment
        if bool(random.getrandbits(1))
        else monotonic_alignment
    )
    aligned_text = text_encoding @ alignment
    features = torch.cat(
        (
            aligned_text,
            F.avg_pool1d(energy.unsqueeze(1), 2),
            F.avg_pool1d(f0.unsqueeze(1), 2),
        ),
        dim=1,
    )
    cropped, _ = _random_crops(
        features,
        batch.mel_lengths.to(features.device) // 2,
    )
    return cropped


def sample_target_prosody_input(
    batch: TrainingBatch,
    current: Tensor,
) -> tuple[Tensor, Tensor]:
    current_lengths = batch.mel_lengths.to(current.device) // 2
    if bool(random.getrandbits(1)):
        return _random_mask_batch(current, current_lengths), current_lengths

    return _random_crops(current, current_lengths)


def _random_crops(values: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    crop_length = int(lengths.min().item()) - 1
    crops = []
    for value, length_value in zip(values, lengths, strict=True):
        length = int(length_value.item())
        start = int(np.random.randint(0, length - crop_length))
        crops.append(value[:, start : start + crop_length])
    result = torch.stack(crops).detach()
    result_lengths = torch.full(
        (values.size(0),), crop_length, dtype=torch.long, device=values.device
    )
    return result, result_lengths


def _random_mask_batch(values: Tensor, lengths: Tensor) -> Tensor:
    minimum = int(lengths.min().item())
    masked = values.clone()
    for index, length_value in enumerate(lengths):
        length = int(length_value.item())
        pieces = int(np.random.randint(0, 5))
        masked_length = 0
        for _ in range(pieces):
            start = int(np.random.randint(0, length))
            end = int(np.random.randint(start, length))
            if end - start + masked_length <= length - minimum:
                masked[index, :, start:end] = 0
                masked_length += end - start
    return masked


def crop_training_batch(
    batch: TrainingBatch,
    aligned_text: Tensor,
    target_f0: Tensor,
    target_energy: Tensor,
    predicted_f0: Tensor,
    predicted_energy: Tensor,
    frames: int,
) -> tuple[Tensor, ...]:
    starts = [
        int(np.random.randint(0, int(length.item() / 2) - frames))
        for length in batch.mel_lengths
    ]
    aligned = torch.stack(
        [
            aligned_text[index, :, start : start + frames]
            for index, start in enumerate(starts)
        ]
    )
    slices = [slice(start * 2, (start + frames) * 2) for start in starts]
    tracks = (
        target_f0,
        target_energy,
        predicted_f0,
        predicted_energy,
    )
    cropped = [
        torch.stack([track[index, item] for index, item in enumerate(slices)])
        for track in tracks
    ]
    cropped_mels = torch.stack(
        [batch.mels[index, :, item] for index, item in enumerate(slices)]
    ).detach()
    waves = [
        batch.waves[index][start * 600 : (start + frames) * 600]
        for index, start in enumerate(starts)
    ]
    waveform = (
        torch.from_numpy(np.stack(waves))
        .to(batch.mels.device)
        .float()
        .unsqueeze(1)
    )
    return (aligned, *cropped, cropped_mels, waveform)


def sample_voice_prompts(batch: TrainingBatch) -> Tensor:
    maximum_frames = int(batch.mel_lengths.min().item() // 2)
    prompt_frames = int(
        np.random.randint((maximum_frames + 1) // 2, maximum_frames + 1)
    )
    prompt_starts = [
        int(np.random.randint(0, int(length.item() / 2) - prompt_frames + 1))
        for length in batch.mel_lengths
    ]
    prompt_slices = [
        slice(start * 2, (start + prompt_frames) * 2)
        for start in prompt_starts
    ]
    prompt_mels = torch.stack(
        [batch.mels[index, :, item] for index, item in enumerate(prompt_slices)]
    ).detach()
    return prompt_mels
