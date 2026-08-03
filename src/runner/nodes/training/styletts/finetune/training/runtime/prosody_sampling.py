import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path


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


def sample_author_prosody_input(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    current: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reproduce the masked/cropped encoder inputs used by small_rec.py."""
    current_lengths = batch.mel_lengths.to(current.device) // 2
    if bool(random.getrandbits(1)):
        return _random_mask_batch(current, current_lengths), current_lengths

    current_crops, current_crop_lengths = _random_crops(current, current_lengths)
    if random.random() < 0.9:
        return current_crops, current_crop_lengths

    reference, reference_lengths = _reference_inputs(runtime, batch)
    return _random_crops(reference, reference_lengths)


@torch.no_grad()
def _reference_inputs(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
) -> tuple[Tensor, Tensor]:
    modules = runtime.models.modules
    device = batch.mels.device
    lengths = batch.reference_mel_lengths.to(device)
    input_lengths = batch.reference_lengths.to(device)
    down_lengths = lengths // (2**runtime.models.n_down)
    mask = length_to_mask(down_lengths, device)
    _, _, alignment = modules.text_aligner(
        batch.reference_mels,
        mask,
        batch.reference_texts,
    )
    alignment = alignment.transpose(-1, -2)[..., 1:].transpose(-1, -2)
    alignment_mask = mask_from_lens(alignment, input_lengths, down_lengths)
    monotonic = maximum_path(alignment, alignment_mask)
    f0, _, _ = modules.pitch_extractor(batch.reference_mels.unsqueeze(1))
    f0 = f0.squeeze(-1)
    energy = log_norm(batch.reference_mels.unsqueeze(1)).squeeze(1)
    full_mask = length_to_mask(lengths, device)
    f0.masked_fill_(full_mask, 0.0)
    energy.masked_fill_(full_mask, 0.0)
    return (
        prosody_inputs(modules.position_embedding, monotonic, f0, energy),
        lengths // 2,
    )


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
