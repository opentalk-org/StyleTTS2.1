from dataclasses import dataclass

from torch import Tensor, nn

from runner.nodes.training.styletts3.testing.styletts_discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
)


@dataclass(frozen=True)
class DiscriminatorEvaluation:
    logits: list[Tensor]
    feature_maps: list[list[Tensor]]


@dataclass(frozen=True)
class StyleTTSDiscriminatorOutput:
    real: DiscriminatorEvaluation
    fake: DiscriminatorEvaluation


class StyleTTSDiscriminators(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.multi_period = MultiPeriodDiscriminator(gradient_checkpointing)
        self.multi_resolution = MultiResSpecDiscriminator(gradient_checkpointing)

    def forward(self, real: Tensor, fake: Tensor) -> StyleTTSDiscriminatorOutput:
        period = self.multi_period(real, fake)
        resolution = self.multi_resolution(real, fake)
        return StyleTTSDiscriminatorOutput(
            real=DiscriminatorEvaluation(
                logits=[*period[0], *resolution[0]],
                feature_maps=[*period[2], *resolution[2]],
            ),
            fake=DiscriminatorEvaluation(
                logits=[*period[1], *resolution[1]],
                feature_maps=[*period[3], *resolution[3]],
            ),
        )


def build_styletts_discriminators(
    gradient_checkpointing: bool = False,
) -> StyleTTSDiscriminators:
    return StyleTTSDiscriminators(gradient_checkpointing)
