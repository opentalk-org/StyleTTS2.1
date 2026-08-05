import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchaudio.models import Conformer

from ..diffusion.modules import Attention, TimePositionalEmbedding


STYLE_TOKEN_COUNT = 50
STYLE_TOKEN_DIM = 512


def length_to_mask(lengths: Tensor) -> Tensor:
    positions = torch.arange(lengths.max(), device=lengths.device)
    return positions.unsqueeze(0) + 1 > lengths.unsqueeze(1)


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256) -> None:
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, values: list[float] | Tensor) -> Tensor:
        if not torch.is_tensor(values):
            values = torch.tensor(values, device=next(self.embedding.parameters()).device)
        shape = values.shape
        embedding = self.embedding(rearrange(values, "... -> (...)"))
        return embedding.view(*shape, self.features)


class TVStyleEncoder(nn.Module):
    """Fixed-length prosody encoder."""

    def __init__(
        self,
        mel_dim: int = 514,
        text_dim: int = STYLE_TOKEN_DIM,
        num_heads: int = 8,
        num_time: int = STYLE_TOKEN_COUNT,
        num_layers: int = 6,
        head_features: int = 64,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=1,
            depthwise_conv_kernel_size=31,
            use_group_norm=True,
        )
        self.conformer_body = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=num_layers - 1,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.cross_attention = Attention(
            features=text_dim,
            num_heads=num_heads,
            head_features=head_features,
            context_features=text_dim,
            use_rel_pos=False,
        )
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, text_dim)
        self.embedder = NumberEmbedder(features=text_dim)

    def forward(self, values: Tensor, input_lengths: Tensor) -> Tensor:
        input_lengths = input_lengths.to(values.device)
        values = values[..., : input_lengths.max()]
        hidden = self.mel_proj(values).transpose(-1, -2)
        hidden, output_lengths = self.conformer_pre(hidden, input_lengths)
        hidden, _ = self.conformer_body(hidden, output_lengths)
        hidden = hidden.transpose(-1, -2)
        indices = torch.arange(self.num_time, device=hidden.device)
        positions = self.positions(indices).unsqueeze(0).expand(hidden.size(0), -1, -1)
        time = torch.arange(hidden.size(-1), device=hidden.device)
        hidden = hidden + self.embedder(time).transpose(-1, -2).expand(hidden.size(0), -1, -1)
        hidden.masked_fill_(length_to_mask(input_lengths).unsqueeze(1), 0.0)
        hidden = self.cross_attention(positions, context=hidden.transpose(-1, -2))
        return hidden.transpose(-1, -2)


class LinearNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, values: Tensor) -> Tensor:
        return self.linear_layer(values)


class DurationPredictor(nn.Module):
    def __init__(self, d_hid: int = 512, nlayers: int = 6, max_dur: int = 50) -> None:
        super().__init__()
        self.transformer = Conformer(
            input_dim=d_hid,
            num_heads=8,
            ffn_dim=d_hid * 2,
            num_layers=nlayers,
            depthwise_conv_kernel_size=7,
            use_group_norm=True,
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)

    def forward(self, text: Tensor, style: Tensor, input_lengths: Tensor, max_size: int) -> Tensor:
        input_lengths = input_lengths.to(text.device)
        text_return = text.new_zeros(text.size(0), text.size(1), max_size)
        text = text[..., : input_lengths.max()]
        style_length = style.size(-1)
        lengths = input_lengths + style_length
        hidden = torch.cat((style, text), dim=-1).transpose(-1, -2)
        hidden, _ = self.transformer(hidden, lengths)
        hidden = hidden.transpose(-1, -2)[:, :, style_length:]
        text_return[:, :, : hidden.size(-1)] = hidden
        return self.duration_proj(text_return.transpose(-1, -2))


class ProsodyPredictor(nn.Module):
    def __init__(self, d_hid: int = 512, nlayers: int = 6, scale_factor: int = 2) -> None:
        super().__init__()
        self.conf_pre = Conformer(
            input_dim=d_hid,
            num_heads=8,
            ffn_dim=d_hid * 2,
            num_layers=nlayers // 2,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.conf_after = Conformer(
            input_dim=d_hid,
            num_heads=8,
            ffn_dim=d_hid * 2,
            num_layers=nlayers // 2,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.F0_proj = LinearNorm(d_hid, 1)
        self.N_proj = LinearNorm(d_hid, 1)
        self.scale_factor = scale_factor

    def forward(self, text: Tensor, style: Tensor, input_lengths: Tensor, max_size: int) -> tuple[Tensor, Tensor]:
        input_lengths = input_lengths.to(text.device)
        text_return = text.new_zeros(text.size(0), text.size(1), max_size * self.scale_factor)
        text = text[..., : input_lengths.max()]
        style_length = style.size(-1)
        lengths = input_lengths + style_length
        hidden = torch.cat((style, text), dim=-1).transpose(-1, -2)
        hidden, _ = self.conf_pre(hidden, lengths)
        hidden = F.interpolate(hidden.transpose(-1, -2), scale_factor=self.scale_factor, mode="nearest").transpose(-1, -2)
        hidden, _ = self.conf_after(hidden, lengths * self.scale_factor)
        hidden = hidden.transpose(-1, -2)[:, :, style_length * self.scale_factor :]
        text_return[:, :, : hidden.size(-1)] = hidden
        features = text_return.transpose(-1, -2)
        f0 = self.F0_proj(features).squeeze(-1)
        return f0, self.N_proj(features).squeeze(-1)


class ProsodyDiscriminator(nn.Module):
    """The author's multimodal prosody discriminator."""

    def __init__(self, mel_dim: int = 514, d_hid: int = 512, nlayers: int = 6, scale_factor: int = 1) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(mel_dim, d_hid, kernel_size=3, padding=1)
        self.d_hid = d_hid
        self.conf_pre = nn.ModuleList(
            Conformer(d_hid, 8, d_hid * 2, 1, 15, use_group_norm=True)
            for _ in range(nlayers // 2)
        )
        self.conf_after = nn.ModuleList(
            Conformer(d_hid, 8, d_hid * 2, 1, 15, use_group_norm=True)
            for _ in range(nlayers // 2)
        )
        self.F0_proj = LinearNorm(d_hid, 1)
        self.sep = nn.Embedding(1, d_hid)
        self.scale_factor = scale_factor

    def forward(
        self,
        text: Tensor,
        style: Tensor,
        input_lengths: Tensor,
        max_size: int,
    ) -> tuple[Tensor, list[Tensor]]:
        input_lengths = input_lengths.to(text.device)
        total_size = max_size * self.scale_factor + style.size(-1) + 1
        text_return = text.new_zeros(text.size(0), self.d_hid, total_size)
        text = self.mel_proj(text[..., : input_lengths.max()])
        lengths = input_lengths + style.size(-1) + 1
        separator = self.sep(
            torch.zeros(text.size(0), dtype=torch.long, device=text.device)
        ).unsqueeze(-1)
        hidden = torch.cat((style, separator, text), dim=-1).transpose(-1, -2)
        feature_maps = []
        for layer in self.conf_pre:
            hidden, _ = layer(hidden, lengths)
            item = text_return.clone()
            item[:, :, : hidden.size(-2)] = hidden.transpose(-1, -2)
            feature_maps.append(item)
        hidden = F.interpolate(
            hidden.transpose(-1, -2),
            scale_factor=self.scale_factor,
            mode="nearest",
        ).transpose(-1, -2)
        for layer in self.conf_after:
            hidden, _ = layer(hidden, lengths * self.scale_factor)
            item = text_return.clone()
            item[:, :, : hidden.size(-2)] = hidden.transpose(-1, -2)
            feature_maps.append(item)
        hidden = hidden.transpose(-1, -2)
        text_return[:, :, : hidden.size(-1)] = hidden
        scores = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        return scores, feature_maps
