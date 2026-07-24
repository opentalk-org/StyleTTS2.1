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
