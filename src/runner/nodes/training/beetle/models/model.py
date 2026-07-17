from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import BeetleConfig
from ..losses.acoustic import MultiResolutionReconstructionLoss
from .modules.audio import (
    AcousticFeatures,
    AudioEncoder,
    AudioPosterior,
    F0Extractor,
    FeatureLinear,
)
from .modules.decoder import Decoder, DecoderOutput
from .modules.discriminators import (
    StyleTTSDiscriminators,
    build_styletts_discriminators,
)
from .modules.generator import Generator


@dataclass(frozen=True)
class Stage1Synthesis:
    posterior: AudioPosterior
    acoustic: AcousticFeatures
    decoded: DecoderOutput
    waveform: Tensor
    sample_mask: Tensor


@dataclass(frozen=True)
class ParameterReport:
    inference: int
    frozen_helper: int
    training_only: int

    @property
    def total(self) -> int:
        return self.inference + self.frozen_helper + self.training_only


def normalized_log_mel_energy(mel: Tensor, frame_mask: Tensor) -> Tensor:
    if mel.ndim != 3 or frame_mask.shape != (mel.shape[0], 1, mel.shape[2]):
        raise ValueError("mel energy requires [B,M,T] mel and [B,1,T] mask")
    numeric_mask = frame_mask[:, 0].to(dtype=mel.dtype)
    energy = mel.mean(dim=1) * numeric_mask
    count = numeric_mask.sum(dim=1, keepdim=True)
    if torch.any(count == 0):
        raise ValueError("each mel item must contain a valid frame")
    mean = energy.sum(dim=1, keepdim=True) / count
    centered = (energy - mean) * numeric_mask
    variance = centered.square().sum(dim=1, keepdim=True) / count
    return centered * torch.rsqrt(variance + 1e-5) * numeric_mask


def _count_unique_parameters(modules: tuple[nn.Module, ...]) -> tuple[int, set[int]]:
    identities: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in identities:
                identities.add(identity)
                count += parameter.numel()
    return count, identities


class Stage1Models(nn.Module):
    def __init__(
        self,
        audio_encoder: AudioEncoder,
        feature_linear: FeatureLinear,
        decoder: Decoder,
        generator: Generator,
        f0_extractor: F0Extractor,
        discriminators: StyleTTSDiscriminators,
        reconstruction_loss: MultiResolutionReconstructionLoss,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.feature_linear = feature_linear
        self.decoder = decoder
        self.generator = generator
        self.f0_extractor = f0_extractor
        self.discriminators = discriminators
        self.reconstruction_loss = reconstruction_loss

    def reconstruct(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        latent_generator: torch.Generator,
        source_generator: torch.Generator,
    ) -> Stage1Synthesis:
        posterior = self.audio_encoder(mel, frame_mask, latent_generator)
        acoustic = self.feature_linear(
            posterior.latent,
            posterior.mask,
            frame_mask,
        )
        decoded = self.decoder(
            posterior.latent,
            acoustic.f0,
            acoustic.n,
            posterior.mask,
            frame_mask,
        )
        waveform = self.generator(
            decoded.features,
            decoded.f0,
            decoded.mask,
            source_generator,
        )
        sample_mask = frame_mask.repeat_interleave(
            self.generator.config.output_hop(),
            dim=-1,
        )
        return Stage1Synthesis(
            posterior=posterior,
            acoustic=acoustic,
            decoded=decoded,
            waveform=waveform,
            sample_mask=sample_mask,
        )

    def acoustic_targets(self, mel: Tensor, frame_mask: Tensor) -> AcousticFeatures:
        return AcousticFeatures(
            f0=self.f0_extractor(mel, frame_mask),
            n=normalized_log_mel_energy(mel, frame_mask),
        )

    def parameter_report(self) -> ParameterReport:
        inference_modules = (
            self.audio_encoder,
            self.feature_linear,
            self.decoder,
            self.generator,
        )
        inference, inference_ids = _count_unique_parameters(inference_modules)
        frozen, frozen_ids = _count_unique_parameters((self.f0_extractor,))
        training, training_ids = _count_unique_parameters((self.discriminators,))
        if (
            inference_ids & frozen_ids
            or inference_ids & training_ids
            or frozen_ids & training_ids
        ):
            raise ValueError("Stage 1 parameter categories must be disjoint")
        return ParameterReport(inference, frozen, training)


def build_stage1_models(
    config: BeetleConfig,
    f0_extractor: F0Extractor,
) -> Stage1Models:
    architecture = config.architecture
    return Stage1Models(
        audio_encoder=AudioEncoder(architecture.posterior),
        feature_linear=FeatureLinear(architecture.feature),
        decoder=Decoder(architecture.decoder),
        generator=Generator(architecture.generator, config.audio.sample_rate),
        f0_extractor=f0_extractor,
        discriminators=build_styletts_discriminators(),
        reconstruction_loss=MultiResolutionReconstructionLoss(
            sample_rate=config.audio.sample_rate,
        ),
    )
