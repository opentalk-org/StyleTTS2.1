import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import TrainingBatch
from ..setup import TrainingRuntime
from .training_forward import ForwardOutput


def style_nuisance_loss(
    runtime: TrainingRuntime,
    output: ForwardOutput,
    batch: TrainingBatch,
    reversal_strength: float,
) -> Tensor:
    factorization = runtime.models.modules.factorization
    parameters = runtime.models.parameters
    content_bag = _phoneme_bag(
        batch.texts,
        batch.input_lengths,
        parameters.n_token,
    )
    return factorization.style_nuisance_loss(
        output.style_target,
        batch.speaker_ids,
        batch.language_ids,
        content_bag,
        reversal_strength,
    )


def _phoneme_bag(texts: Tensor, lengths: Tensor, content_dim: int) -> Tensor:
    positions = torch.arange(texts.size(1), device=texts.device)
    valid = positions[None, :] < lengths.to(texts.device)[:, None]
    tokens = F.one_hot(texts, num_classes=content_dim).to(torch.float32)
    return (tokens * valid.unsqueeze(-1)).amax(1)
