from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import BeetleConfig
from ..losses.acoustic import MultiResolutionReconstructionLoss
from .acoustic import log_mel_l2_energy
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
from .parameters import ParameterReport, count_unique_parameters


@dataclass(frozen=True)
class AcousticSynthesis:
    posterior: AudioPosterior
    acoustic: AcousticFeatures
    decoded: DecoderOutput
    waveform: Tensor
    sample_mask: Tensor


class AcousticModels(nn.Module):
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
        self.output_hop = generator.config.output_hop()
        self.latent_downsample_rate = audio_encoder.config.downsample_rate
        self.encoder_context_frames = audio_encoder.config.receptive_field_mel_frames()

    def reconstruct(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        latent_generator: torch.Generator,
    ) -> AcousticSynthesis:
        posterior = self.audio_encoder(mel, frame_mask, latent_generator)
        acoustic = self.feature_linear(
            posterior.latent,
            posterior.mask,
            frame_mask,
        )
        return self._render(
            posterior,
            acoustic,
            frame_mask,
        )

    def _render(
        self,
        posterior: AudioPosterior,
        acoustic: AcousticFeatures,
        frame_mask: Tensor,
    ) -> AcousticSynthesis:
        decoded = self.decoder(
            posterior.latent,
            posterior.mask,
            frame_mask,
        )
        waveform = self.generator(
            decoded.features,
            decoded.mask,
        )
        sample_mask = frame_mask.repeat_interleave(
            self.output_hop,
            dim=-1,
        )
        return AcousticSynthesis(
            posterior=posterior,
            acoustic=acoustic,
            decoded=decoded,
            waveform=waveform,
            sample_mask=sample_mask,
        )

    def acoustic_targets(self, mel: Tensor, frame_mask: Tensor) -> AcousticFeatures:
        return AcousticFeatures(
            f0=self.f0_target(mel, frame_mask),
            n=self.n_target(mel, frame_mask),
        )

    def f0_target(self, mel: Tensor, frame_mask: Tensor) -> Tensor:
        return self.f0_extractor(mel, frame_mask)

    def n_target(self, mel: Tensor, frame_mask: Tensor) -> Tensor:
        return log_mel_l2_energy(mel, frame_mask)

    def parameter_report(self) -> ParameterReport:
        inference_modules = (
            self.audio_encoder,
            self.feature_linear,
            self.decoder,
            self.generator,
        )
        inference, inference_ids = count_unique_parameters(inference_modules)
        frozen, frozen_ids = count_unique_parameters((self.f0_extractor,))
        training, training_ids = count_unique_parameters((self.discriminators,))
        return ParameterReport(inference, frozen, training)


def build_acoustic_models(
    config: BeetleConfig,
    f0_extractor: F0Extractor,
) -> AcousticModels:
    architecture = config.architecture
    return AcousticModels(
        audio_encoder=AudioEncoder(architecture.posterior),
        feature_linear=FeatureLinear(architecture.feature),
        decoder=Decoder(architecture.decoder),
        generator=Generator(architecture.generator),
        f0_extractor=f0_extractor,
        discriminators=build_styletts_discriminators(),
        reconstruction_loss=MultiResolutionReconstructionLoss(
            sample_rate=config.audio.sample_rate,
            mel_channels=config.audio.mel_channels,
            complex_reconstruction_steps=(
                config.training.complex_reconstruction_steps
            ),
        ),
    )
