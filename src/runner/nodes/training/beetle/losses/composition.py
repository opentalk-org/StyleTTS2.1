from dataclasses import dataclass

from torch import Tensor

from ..models.bundle import Stage1Models, Stage1Synthesis
from ..models.features import AcousticFeatures
from .acoustic import masked_f0_mse, masked_kl_standard_normal, masked_n_mse
from .adversarial import discriminator_step_loss, generator_step_loss


@dataclass(frozen=True)
class Stage1LossWeights:
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
            raise ValueError("Stage 1 loss weights must be nonnegative")


@dataclass(frozen=True)
class Stage1Losses:
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

    def discriminator_total(self, weights: Stage1LossWeights) -> Tensor:
        return self.discriminator * weights.discriminator

    def generator_total(self, weights: Stage1LossWeights) -> Tensor:
        return (
            self.encoder_kl * weights.encoder_kl
            + self.f0 * weights.f0
            + self.n * weights.n
            + self.reconstruction * weights.reconstruction
            + self.generator_adversarial * weights.generator_adversarial
            + self.feature_matching * weights.feature_matching
        )


def compute_stage1_losses(
    models: Stage1Models,
    synthesis: Stage1Synthesis,
    targets: AcousticFeatures,
    target_waveform: Tensor,
) -> Stage1Losses:
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
    return Stage1Losses(
        encoder_kl=masked_kl_standard_normal(
            synthesis.posterior.mean,
            synthesis.posterior.log_scale,
            synthesis.posterior.mask,
        ),
        f0=masked_f0_mse(
            synthesis.acoustic.f0,
            targets.f0,
            synthesis.decoded.mask,
        ),
        n=masked_n_mse(
            synthesis.acoustic.n,
            targets.n,
            synthesis.decoded.mask,
        ),
        reconstruction=reconstruction.total,
        discriminator=discriminator,
        generator_adversarial=generator.adversarial,
        feature_matching=generator.feature_matching,
    )
