import numpy as np
import torch
from torch import Tensor

from ..data import TrainingBatch


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
        [aligned_text[index, :, start : start + frames] for index, start in enumerate(starts)]
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
    waves = [
        batch.waves[index][start * 600 : (start + frames) * 600]
        for index, start in enumerate(starts)
    ]
    waveform = torch.from_numpy(np.stack(waves)).to(batch.mels.device).float().unsqueeze(1)
    return (aligned, *cropped, waveform)
