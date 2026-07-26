from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from ...config.architecture import DecoderConfig
from .convolution import MaskedResidualBlock


@dataclass(frozen=True)
class DecoderOutput:
    features: Tensor
    mask: Tensor


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_upsample = weight_norm(
            nn.ConvTranspose1d(
                config.latent_channels,
                config.generator_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.refinement = nn.ModuleList(
            MaskedResidualBlock(
                config.generator_channels,
                kernel_size=3,
                dilation=3**index,
                dropout=config.dropout,
            )
            for index in range(config.decode_block_count)
        )

    def forward(
        self,
        latent: Tensor,
        latent_mask: Tensor,
        frame_mask: Tensor,
    ) -> DecoderOutput:
        batch_size, _, latent_frames = latent.shape
        frame_frames = latent_frames * 2
        boolean_latent_mask = latent_mask.to(dtype=torch.bool)
        boolean_frame_mask = frame_mask.to(dtype=torch.bool)
        numeric_latent_mask = boolean_latent_mask.to(dtype=latent.dtype)
        numeric_frame_mask = boolean_frame_mask.to(dtype=latent.dtype)
        masked_latent = latent * numeric_latent_mask
        features = self.latent_upsample(masked_latent)
        features = features * numeric_frame_mask
        for block in self.refinement:
            features = block(features, numeric_frame_mask)
        return DecoderOutput(
            features * numeric_frame_mask,
            boolean_frame_mask,
        )
