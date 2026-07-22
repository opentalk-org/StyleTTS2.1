from dataclasses import dataclass

from torch import Tensor

from ..models.model import AcousticModels, AcousticSynthesis
from ..models.modules.audio import AcousticFeatures
from .acoustic import (
    masked_f0_smooth_l1,
    masked_kl_standard_normal,
    masked_n_smooth_l1,
)
from .adversarial import discriminator_step_loss, generator_step_loss
from .conditional import (
    ConditionalLossInput,
    ConditionalLossOutput,
    ConditionalLossWeights,
    compute_conditional_losses,
)

__all__ = [
    "AcousticLosses",
    "AcousticLossWeights",
    "ConditionalLossInput",
    "ConditionalLossOutput",
    "ConditionalLossWeights",
    "compute_acoustic_losses",
    "compute_conditional_losses",
]


@dataclass(frozen=True)
class AcousticLossWeights:
    encoder_kl: float
    f0: float
    n: float
    reconstruction: float
    discriminator: float
    generator_adversarial: float
    feature_matching: float

    def __post_init__(self) -> None:
        values = (
            self.encoder_kl,
            self.f0,
            self.n,
            self.reconstruction,
            self.discriminator,
            self.generator_adversarial,
            self.feature_matching,
        )
        if any(value < 0 for value in values):
            raise ValueError("acoustic loss weights must be nonnegative")


@dataclass(frozen=True)
class AcousticLosses:
    encoder_kl: Tensor
    f0: Tensor
    n: Tensor
    reconstruction: Tensor
    discriminator: Tensor
    generator_adversarial: Tensor
    feature_matching: Tensor

    def named(self) -> dict[str, Tensor]:
        return {
            "encoder_kl": self.encoder_kl,
            "f0": self.f0,
            "n": self.n,
            "reconstruction": self.reconstruction,
            "discriminator": self.discriminator,
            "generator_adversarial": self.generator_adversarial,
            "feature_matching": self.feature_matching,
        }

    def discriminator_total(self, weights: AcousticLossWeights) -> Tensor:
        return self.discriminator * weights.discriminator

    def generator_total(self, weights: AcousticLossWeights) -> Tensor:
        return (
            self.encoder_kl * weights.encoder_kl
            + self.f0 * weights.f0
            + self.n * weights.n
            + self.reconstruction * weights.reconstruction
            + self.generator_adversarial * weights.generator_adversarial
            + self.feature_matching * weights.feature_matching
        )


def compute_acoustic_losses(
    models: AcousticModels,
    synthesis: AcousticSynthesis,
    targets: AcousticFeatures,
    target_waveform: Tensor,
) -> AcousticLosses:
    if target_waveform.shape != synthesis.waveform.shape:
        raise ValueError("target and generated waveforms must have equal shapes")
    reconstruction = models.reconstruction_loss(
        synthesis.waveform,
        target_waveform,
        synthesis.sample_mask,
    )
    discriminator = discriminator_step_loss(
        models.discriminators,
        target_waveform,
        synthesis.waveform,
    )
    generator = generator_step_loss(
        models.discriminators,
        target_waveform,
        synthesis.waveform,
    )
    return AcousticLosses(
        encoder_kl=masked_kl_standard_normal(
            synthesis.posterior.mean,
            synthesis.posterior.log_scale,
            synthesis.posterior.mask,
        ),
        f0=masked_f0_smooth_l1(
            synthesis.acoustic.f0,
            targets.f0,
            synthesis.decoded.mask,
        ),
        n=masked_n_smooth_l1(
            synthesis.acoustic.n,
            targets.n,
            synthesis.decoded.mask,
        ),
        reconstruction=reconstruction.total,
        discriminator=discriminator,
        generator_adversarial=generator.adversarial,
        feature_matching=generator.feature_matching,
    )
