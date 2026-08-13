from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm


@dataclass(frozen=True)
class ProsodyLatents:
    continuous: Tensor
    quantized: Tensor
    continuous_style: Tensor
    quantized_style: Tensor
    indices: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        weight_norm(nn.Conv1d(input_dim, output_dim, kernel_size=1)),
        nn.LeakyReLU(0.1),
    )


class VectorQuantizer(nn.Module):
    """StyleTTS-ZS normalized learned vector-codebook quantizer."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.in_proj = _projection(input_dim, codebook_dim)
        self.out_proj = _projection(codebook_dim, input_dim)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self.in_proj(values)
        flattened = encoded.transpose(1, 2).reshape(-1, encoded.size(1))
        normalized_values = F.normalize(flattened)
        normalized_codebook = F.normalize(self.codebook.weight)
        distances = (
            normalized_values.square().sum(1, keepdim=True)
            - 2 * normalized_values @ normalized_codebook.t()
            + normalized_codebook.square().sum(1).unsqueeze(0)
        )
        indices = distances.argmin(1).reshape(values.size(0), values.size(2))
        quantized = self.codebook(indices).transpose(1, 2)
        commitment = F.mse_loss(encoded, quantized.detach(), reduction="none").mean((1, 2))
        codebook = F.mse_loss(quantized, encoded.detach(), reduction="none").mean((1, 2))
        straight_through = encoded + (quantized - encoded).detach()
        return self.out_proj(straight_through), commitment, codebook, indices

    def embed(self, indices: Tensor) -> Tensor:
        values = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        return self.out_proj(values)


class ResidualVectorQuantizer(nn.Module):
    """The nine-stage learned RVQ used by StyleTTS-ZS."""

    def __init__(
        self,
        input_dim: int,
        stages: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = input_dim
        self.num_stages = stages
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            VectorQuantizer(input_dim, codebook_size, codebook_dim)
            for _ in range(stages)
        )

    def forward(self, values: Tensor) -> ProsodyLatents:
        quantized = torch.zeros_like(values)
        residual = values
        commitments = values.new_zeros(())
        codebook_losses = values.new_zeros(())
        indices = []
        for quantizer in self.quantizers:
            stage, commitment, codebook, stage_indices = quantizer(residual)
            quantized = quantized + stage
            residual = residual - stage
            commitments = commitments + commitment.mean()
            codebook_losses = codebook_losses + codebook.mean()
            indices.append(stage_indices)
        return ProsodyLatents(
            continuous=values,
            quantized=quantized,
            continuous_style=values,
            quantized_style=quantized,
            indices=torch.stack(indices, dim=1),
            commitment_loss=commitments,
            codebook_loss=codebook_losses,
        )

    def decode_continuous(self, latent: Tensor) -> Tensor:
        return latent

    def decode(self, indices: Tensor) -> Tensor:
        quantizers = [cast(VectorQuantizer, item) for item in self.quantizers]
        quantized = torch.zeros_like(quantizers[0].embed(indices[:, 0]))
        for quantizer, stage_indices in zip(
            quantizers,
            indices.unbind(1),
            strict=True,
        ):
            quantized = quantized + quantizer.embed(stage_indices)
        return quantized
