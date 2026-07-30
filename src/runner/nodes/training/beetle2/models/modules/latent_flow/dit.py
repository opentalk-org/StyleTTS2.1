import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScalarEmbedding(nn.Module):
    def __init__(self, frequency_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.frequency_channels = frequency_channels
        self.layers = nn.Sequential(
            nn.Linear(frequency_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, values: Tensor) -> Tensor:
        half = self.frequency_channels // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=values.device, dtype=torch.float32)
            / half
        )
        angles = values.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.layers(embedding.to(dtype=values.dtype))


class Attention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_channels = channels // heads
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.projection = nn.Linear(channels, channels)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        batch, tokens, channels = features.shape
        qkv = self.qkv(features).view(
            batch,
            tokens,
            3,
            self.heads,
            self.head_channels,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask[:, None, None, :],
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection(attended)


class DiTBlock(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_channels = int(channels * mlp_ratio)
        self.condition_input = nn.Linear(channels * 2, channels)
        self.attention_norm = nn.LayerNorm(
            channels,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attention = Attention(channels, heads)
        self.mlp_norm = nn.LayerNorm(
            channels,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_channels),
            nn.GELU(),
            nn.Linear(mlp_channels, channels),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, channels * 6),
        )

    def forward(self, features: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.unsqueeze(-1).to(dtype=features.dtype)
        features = self.condition_input(
            torch.cat((features, condition), dim=-1)
        ) * numeric_mask
        values = self.modulation(condition).chunk(6, dim=-1)
        attention_input = modulate(self.attention_norm(features), values[0], values[1])
        features = features + values[2] * self.attention(attention_input, mask)
        mlp_input = modulate(self.mlp_norm(features), values[3], values[4])
        features = features + values[5] * self.mlp(mlp_input)
        return features * numeric_mask


class FinalLayer(nn.Module):
    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            channels,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, channels * 2),
        )
        self.output = nn.Linear(channels, output_channels)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.output(modulate(self.norm(features), shift, scale))


def modulate(features: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return features * (1 + scale) + shift


def position_embedding(
    token_count: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    half = channels // 2
    positions = torch.arange(token_count, device=device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=device, dtype=torch.float32)
        / half
    )
    angles = positions[:, None] * frequencies[None]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1).to(dtype=dtype)
