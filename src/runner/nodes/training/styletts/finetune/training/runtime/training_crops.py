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


def sample_voice_prompts(batch: TrainingBatch) -> Tensor:
    prompt_frames = int(batch.mel_lengths.min().item() / 2 - 1)
    prompt_starts = [
        int(np.random.randint(0, int(length.item() / 2) - prompt_frames))
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
