"""Patch-based 1D DiT-S/2 latent velocity model."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ....config.architecture import LatentFlowConfig
from ..conditioning import ProjectedConditions
from .dit import DiTBlock, FinalLayer, ScalarEmbedding, position_embedding
from .sampling import FlowTrainingSample, patch_mask, sample_flow_training_case

__all__ = [
    "FlowTrainingSample",
    "LatentFlowModel",
    "sample_flow_training_case",
]


class LatentFlowModel(nn.Module):
    def __init__(self, config: LatentFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.latent_channels,
            config.hidden_channels,
            config.patch_size,
            stride=config.patch_size,
        )
        self.condition_projection = nn.Conv1d(
            config.condition_channels,
            config.hidden_channels,
            config.patch_size,
            stride=config.patch_size,
        )
        self.start_time_embedding = ScalarEmbedding(
            config.time_embedding_channels,
            config.hidden_channels,
        )
        self.end_time_embedding = ScalarEmbedding(
            config.time_embedding_channels,
            config.hidden_channels,
        )
        self.blocks = nn.ModuleList(
            DiTBlock(
                config.hidden_channels,
                config.attention_heads,
                config.mlp_ratio,
            )
            for _ in range(config.layer_count)
        )
        self.final = FinalLayer(
            config.hidden_channels,
            config.patch_size * config.latent_channels,
        )
        self.apply(_initialize_projection)
        self._initialize_time_embeddings()

    def forward(
        self,
        state: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        conditions: ProjectedConditions,
        mask: Tensor,
    ) -> Tensor:
        frame_count = state.shape[-1]
        padding = (-frame_count) % self.config.patch_size
        numeric_mask = mask.to(dtype=state.dtype)
        padded_state = F.pad(state * numeric_mask, (0, padding))
        padded_condition = F.pad(
            conditions.concatenated() * numeric_mask,
            (0, padding),
        )
        token_mask = patch_mask(mask, self.config.patch_size)[:, 0]
        latent_features = self.input_projection(padded_state).transpose(1, 2)
        condition_features = self.condition_projection(
            padded_condition
        ).transpose(1, 2)
        features = latent_features
        start_time_tokens = start_time[:, 0, :: self.config.patch_size]
        end_time_tokens = end_time[:, 0, :: self.config.patch_size].to(
            dtype=state.dtype
        )
        condition = (
            condition_features
            + self.start_time_embedding(start_time_tokens)
            + self.end_time_embedding(end_time_tokens)
        )
        features = features + position_embedding(
            features.shape[1],
            features.shape[2],
            features.device,
            features.dtype,
        )
        features = features * token_mask.unsqueeze(-1)
        for block in self.blocks:
            features = block(features, condition, token_mask)
        patches = self.final(features, condition)
        output = patches.view(
            state.shape[0],
            patches.shape[1],
            self.config.patch_size,
            self.config.latent_channels,
        )
        output = output.permute(0, 3, 1, 2).reshape(state.shape[0], state.shape[1], -1)
        return output[..., :frame_count] * numeric_mask

    def _initialize_time_embeddings(self) -> None:
        for embedding in (self.start_time_embedding, self.end_time_embedding):
            nn.init.normal_(embedding.layers[0].weight, std=0.02)
            nn.init.normal_(embedding.layers[2].weight, std=0.02)


def _initialize_projection(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
