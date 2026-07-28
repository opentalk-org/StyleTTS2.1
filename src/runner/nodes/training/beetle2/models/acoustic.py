import torch
from torch import Tensor


def log_mel_l2_energy(mel: Tensor, frame_mask: Tensor) -> Tensor:
    numeric_mask = frame_mask[:, 0].to(dtype=mel.dtype)
    linear_mel = torch.exp(mel)
    return torch.log(torch.linalg.vector_norm(linear_mel, dim=1)) * numeric_mask
