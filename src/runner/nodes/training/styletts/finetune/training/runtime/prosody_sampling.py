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
    """Sample masked or cropped prosody solely from the target utterance."""
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
