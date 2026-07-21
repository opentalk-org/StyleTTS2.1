import torch
from torch import Tensor


def log_mel_l2_energy(mel: Tensor, frame_mask: Tensor) -> Tensor:
    if mel.ndim != 3 or frame_mask.shape != (mel.shape[0], 1, mel.shape[2]):
        raise ValueError("mel energy requires [B,M,T] mel and [B,1,T] mask")
    numeric_mask = frame_mask[:, 0].to(dtype=mel.dtype)
    linear_mel = torch.exp(mel)
    return torch.log(torch.linalg.vector_norm(linear_mel, dim=1)) * numeric_mask
