import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm, weight_norm

from ..decoder_blocks import UpSample1d
from ..utils import get_padding, init_weights


def resize_features(features: Tensor, length: int) -> Tensor:
    return F.interpolate(features, size=length, mode="linear", align_corners=False)


class AdditiveResBlock1d(nn.Module):
    """Author decoder residual topology with AdaIN replaced by local addition."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        feature_dim: int,
        upsample: str = "none",
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        # Keep a fixed affine after normalization so the pretrained decoder's
        # AdaIN value at zero style can be preserved during conversion.  These
        # parameters are not conditioned normalization; local conditioning is
        # still supplied only by ``feature_linear``.
        self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm1d(dim_out, affine=True)
        self.feature_linear = nn.Conv1d(feature_dim, dim_in, 1)
        self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, bias=False)) if self.learned_sc else nn.Identity()
        self.dropout = nn.Dropout(dropout_p)
        self.pool = (
            nn.Identity()
            if upsample == "none"
            else weight_norm(nn.ConvTranspose1d(dim_in, dim_in, 3, stride=2, groups=dim_in, padding=1, output_padding=1))
        )
        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        nn.init.normal_(self.feature_linear.weight, std=1e-4)
        nn.init.zeros_(self.feature_linear.bias)

    def forward(self, values: Tensor, features: Tensor) -> Tensor:
        shortcut = self.upsample(values)
        if self.learned_sc:
            shortcut = self.conv1x1(shortcut)
        hidden = self.norm1(values)
        hidden = hidden + self.feature_linear(resize_features(features, hidden.size(-1)))
        hidden = self.pool(F.leaky_relu(hidden, 0.2))
        hidden = self.conv1(self.dropout(hidden))
        hidden = self.norm2(hidden)
        hidden = self.conv2(self.dropout(F.leaky_relu(hidden, 0.2)))
        return (hidden + shortcut) / math.sqrt(2)


class AdditiveGeneratorBlock(nn.Module):
    """Author Snake residual block with pointwise local conditioning."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: list[int],
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=item, padding=get_padding(kernel_size, item)))
            for item in dilation
        )
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        )
        self.norm1 = nn.ModuleList(nn.InstanceNorm1d(channels, affine=True) for _ in dilation)
        self.norm2 = nn.ModuleList(nn.InstanceNorm1d(channels, affine=True) for _ in dilation)
        self.feature_linear = nn.ModuleList(
            nn.Conv1d(feature_dim, channels, 1) for _ in dilation
        )
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation)
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        for adapter in self.feature_linear:
            nn.init.normal_(adapter.weight, std=1e-4)
            nn.init.zeros_(adapter.bias)

    def forward(self, values: Tensor, features: Tensor) -> Tensor:
        items = zip(
            self.convs1,
            self.convs2,
            self.norm1,
            self.norm2,
            self.feature_linear,
            self.alpha1,
            self.alpha2,
            strict=True,
        )
        for conv1, conv2, norm1, norm2, linear, alpha1, alpha2 in items:
            local = resize_features(features, values.size(-1))
            hidden = norm1(values) + linear(local)
            hidden = hidden + torch.sin(alpha1 * hidden).square() / alpha1
            hidden = conv1(hidden)
            hidden = norm2(hidden)
            hidden = hidden + torch.sin(alpha2 * hidden).square() / alpha2
            values = values + conv2(hidden)
        return values

    def remove_weight_norm(self) -> None:
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class LocalDecoderBackbone(nn.Module):
    def __init__(
        self,
        content_dim: int = 512,
        voice_dim: int = 128,
        language_dim: int = 32,
    ) -> None:
        super().__init__()
        self.content_condition = nn.Conv1d(content_dim, 32, 1)
        self.voice_condition = nn.Conv1d(voice_dim, 64, 1)
        self.language_condition = nn.Conv1d(language_dim, 16, 1)
        self.prosody_condition = nn.Conv1d(2, 16, 1)
        self.feature_dim = 128
        self.encode = AdditiveResBlock1d(content_dim + 2, 1024, self.feature_dim)
        self.decode = nn.ModuleList(
            (
                AdditiveResBlock1d(1024 + 2 + 64, 1024, self.feature_dim),
                AdditiveResBlock1d(1024 + 2 + 64, 1024, self.feature_dim),
                AdditiveResBlock1d(1024 + 2 + 64, 1024, self.feature_dim),
                AdditiveResBlock1d(1024 + 2 + 64, 512, self.feature_dim, upsample="half"),
            )
        )
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(content_dim, 64, kernel_size=1)))

    def forward(
        self,
        content: Tensor,
        f0: Tensor,
        energy: Tensor,
        voice: Tensor,
        language: Tensor,
        frame_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = self._feature_track(content, f0, energy, voice, language, frame_mask)
        f0, energy = self._prepare_inputs(f0, energy)
        f0_half = self.F0_conv(f0.unsqueeze(1))
        energy_half = self.N_conv(energy.unsqueeze(1))
        hidden = self.encode(torch.cat((content, f0_half, energy_half), dim=1), features)
        content_residual = self.asr_res(content)
        for block in self.decode:
            hidden = torch.cat((hidden, content_residual, f0_half, energy_half), dim=1)
            hidden = block(hidden, features)
        return hidden, f0, features

    def _prepare_inputs(self, f0: Tensor, energy: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training:
            return f0, energy
        kernel_size = (0, 3, 7, 15)[random.randint(0, 3)]
        if kernel_size:
            kernel = torch.ones(1, 1, kernel_size, device=energy.device, dtype=energy.dtype)
            energy = F.conv1d(energy.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1) / kernel_size
        return f0, energy

    def _feature_track(
        self,
        content: Tensor,
        f0: Tensor,
        energy: Tensor,
        voice: Tensor,
        language: Tensor,
        frame_mask: Tensor | None,
    ) -> Tensor:
        length = f0.size(-1)
        content_track = self.content_condition(resize_features(content, length))
        voice_track = self._expand_track(voice, length)
        language_track = self._expand_track(language, length)
        voice_track = self.voice_condition(voice_track)
        language_track = self.language_condition(language_track)
        prosody_track = self.prosody_condition(
            torch.stack((f0, energy), dim=1)
        )
        features = torch.cat(
            (content_track, voice_track, language_track, prosody_track),
            dim=1,
        )
        if frame_mask is not None:
            features = features * frame_mask[:, None, :length]
        return features

    @staticmethod
    def _expand_track(values: Tensor, length: int) -> Tensor:
        if values.ndim == 2:
            return values.unsqueeze(-1).expand(-1, -1, length)
        return resize_features(values, length)
