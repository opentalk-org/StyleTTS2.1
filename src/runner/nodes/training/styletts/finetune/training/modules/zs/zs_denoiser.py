import torch
from torch import Tensor, nn
from torchaudio.models import Conformer

from ..diffusion.modules import FixedEmbedding, TimePositionalEmbedding
from ..diffusion.utils import rand_bool
from .zs_prosody import NumberEmbedder


class StyleDiffuser(nn.Module):
    """The author's StyleTTS-ZS conditional Conformer, adapted to interval time."""

    def __init__(
        self,
        mel_dim: int = 512,
        text_dim: int = 768,
        style_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        embedding_max_length: int = 512,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.feature_proj = nn.Conv1d(514, text_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                Conformer(
                    input_dim=text_dim,
                    num_heads=num_heads,
                    ffn_dim=text_dim * 2,
                    num_layers=1,
                    depthwise_conv_kernel_size=15 if index == 0 else 7,
                    use_group_norm=True,
                )
                for index in range(num_layers)
            ]
        )
        self.out = nn.Conv1d(text_dim, style_dim, 1)
        self.to_time = self._time_embedding(text_dim)
        self.to_start_time = self._time_embedding(text_dim)
        self.fixed_embedding = FixedEmbedding(embedding_max_length, text_dim)
        self.fixed_feature = FixedEmbedding(embedding_max_length * 4, 514)
        self.embedder = NumberEmbedder(features=text_dim)
        self.sep = nn.Embedding(num_embeddings=2, embedding_dim=text_dim)

    @staticmethod
    def _time_embedding(text_dim: int) -> nn.Sequential:
        return nn.Sequential(
            TimePositionalEmbedding(dim=text_dim, out_features=text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
        )

    def run(
        self,
        values: Tensor,
        r: Tensor,
        t: Tensor,
        embedding: Tensor,
        features: Tensor,
        input_lengths: Tensor,
    ) -> Tensor:
        input_lengths = input_lengths.to(values.device)
        mapping = self.to_time(t) + self.to_start_time(r)
        mel = self.mel_proj(values)
        positions = self.embedder(torch.arange(mel.size(-1), device=mel.device))
        mel = mel + positions.transpose(-1, -2).expand(mel.size(0), -1, -1)
        text = embedding.transpose(-1, -2)[..., : input_lengths.max()]
        features = self.feature_proj(features)
        feature_separator = self.sep(torch.zeros(values.size(0), dtype=torch.long, device=values.device)).unsqueeze(-1)
        text = torch.cat((features, feature_separator, text), dim=-1)
        mel_length = mel.size(-1)
        lengths = input_lengths + mel_length + features.size(-1) + 2
        value_separator = self.sep(torch.ones(values.size(0), dtype=torch.long, device=values.device)).unsqueeze(-1)
        hidden = torch.cat((mel, value_separator, text), dim=-1).transpose(-1, -2)
        mapping = mapping.unsqueeze(1).expand(-1, hidden.size(1), -1)
        for block in self.blocks:
            hidden = hidden + mapping
            hidden, lengths = block(hidden, lengths)
        return self.out(hidden.transpose(-1, -2)[:, :, :mel_length])

    def forward(
        self,
        values: Tensor,
        r: Tensor,
        t: Tensor,
        input_lengths: Tensor,
        embedding: Tensor,
        features: Tensor,
        embedding_mask_proba: float = 0.0,
        embedding_scale: float = 1.0,
        feature_scale: float = 1.0,
    ) -> Tensor:
        batch, device = embedding.size(0), embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        fixed_features = self.fixed_feature(features.transpose(-1, -2)).transpose(-1, -2)
        if embedding_mask_proba > 0:
            embedding_mask = rand_bool((batch, 1, 1), embedding_mask_proba, device)
            feature_mask = rand_bool((batch, 1, 1), embedding_mask_proba, device)
            embedding = torch.where(embedding_mask, fixed_embedding, embedding)
            features = torch.where(feature_mask, fixed_features, features)
        if embedding_scale == 1.0 and feature_scale == 1.0:
            return self.run(values, r, t, embedding, features, input_lengths)
        conditioned = self.run(values, r, t, embedding, features, input_lengths)
        unconditioned = self.run(values, r, t, fixed_embedding, fixed_features, input_lengths)
        text_only = self.run(values, r, t, embedding, fixed_features, input_lengths)
        feature_only = self.run(values, r, t, fixed_embedding, features, input_lengths)
        return (
            unconditioned
            + (text_only - unconditioned) * embedding_scale
            + (feature_only - unconditioned) * feature_scale
            + (conditioned - text_only - feature_only + unconditioned)
        )
