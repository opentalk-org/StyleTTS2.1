from dataclasses import dataclass

__all__ = [
    "AcousticLossWeights",
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
