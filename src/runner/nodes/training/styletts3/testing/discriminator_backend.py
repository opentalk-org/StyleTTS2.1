from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from runner.nodes.training.styletts3.testing.styletts_discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
)
from runner.nodes.training.styletts3.testing.wave_u_net_discriminator import (
    WaveUNetDiscriminator,
)


class DiscriminatorBackend(str, Enum):
    WAVE_UNET = "wave_unet"
    STYLETTS = "styletts"


@dataclass(frozen=True)
class DiscriminatorEvaluation:
    logits: list[torch.Tensor]
    feature_maps: list[list[torch.Tensor]]


class VocoderDiscriminator(nn.Module, ABC):
    use_autocast = True

    @abstractmethod
    def evaluate(self, waveform: torch.Tensor) -> DiscriminatorEvaluation:
        raise NotImplementedError

    def evaluate_pair(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> tuple[DiscriminatorEvaluation, DiscriminatorEvaluation]:
        batch_size = real.shape[0]
        combined = self.evaluate(torch.cat([real, fake], dim=0))
        return _split_evaluation(combined, batch_size)

    def discriminator_loss(
        self,
        real: DiscriminatorEvaluation,
        fake: DiscriminatorEvaluation,
    ) -> torch.Tensor:
        losses = [
            (real_logit - 1.0).pow(2).mean() + fake_logit.pow(2).mean()
            for real_logit, fake_logit in zip(real.logits, fake.logits)
        ]
        return torch.stack(losses).sum()

    def generator_adv_loss(
        self,
        real: DiscriminatorEvaluation,
        fake: DiscriminatorEvaluation,
    ) -> torch.Tensor:
        del real
        return torch.stack([(logit - 1.0).pow(2).mean() for logit in fake.logits]).sum()

    def feature_matching_loss(
        self,
        real: DiscriminatorEvaluation,
        fake: DiscriminatorEvaluation,
    ) -> torch.Tensor:
        losses = [
            (real_map - fake_map).abs().mean()
            for real_group, fake_group in zip(real.feature_maps, fake.feature_maps)
            for real_map, fake_map in zip(real_group, fake_group)
        ]
        return torch.stack(losses).sum()


class WaveUNetBackend(VocoderDiscriminator):
    def __init__(self) -> None:
        super().__init__()
        self.discriminator = WaveUNetDiscriminator()

    def evaluate(self, waveform: torch.Tensor) -> DiscriminatorEvaluation:
        logits, maps = self.discriminator(waveform)
        return DiscriminatorEvaluation([logits], [maps])


class RelativeStyleTTSBackend(VocoderDiscriminator, ABC):
    def discriminator_loss(
        self,
        real: DiscriminatorEvaluation,
        fake: DiscriminatorEvaluation,
    ) -> torch.Tensor:
        return super().discriminator_loss(real, fake) + _relative_loss(real.logits, fake.logits)

    def generator_adv_loss(
        self,
        real: DiscriminatorEvaluation,
        fake: DiscriminatorEvaluation,
    ) -> torch.Tensor:
        return super().generator_adv_loss(real, fake) + _relative_loss(real.logits, fake.logits)


class StyleTTSBackend(RelativeStyleTTSBackend):
    use_autocast = False

    def __init__(self) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.mrsd = MultiResSpecDiscriminator()

    def evaluate(self, waveform: torch.Tensor) -> DiscriminatorEvaluation:
        discriminators = (*self.mpd.discriminators, *self.mrsd.discriminators)
        logits: list[torch.Tensor] = []
        maps: list[list[torch.Tensor]] = []
        for discriminator in discriminators:
            logit, feature_map = discriminator(waveform)
            logits.append(logit)
            maps.append(feature_map)
        return DiscriminatorEvaluation(logits, maps)


def _relative_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for real, fake in zip(real_logits, fake_logits):
        difference = real - fake
        median = torch.median(difference)
        selected = (difference - median).pow(2)[real < fake + median]
        relative = selected.mean()
        losses.append(0.04 - torch.relu(relative.new_tensor(0.04) - relative))
    return torch.stack(losses).sum()


def _split_evaluation(
    combined: DiscriminatorEvaluation,
    batch_size: int,
) -> tuple[DiscriminatorEvaluation, DiscriminatorEvaluation]:
    real_logits = [logit[:batch_size] for logit in combined.logits]
    fake_logits = [logit[batch_size:] for logit in combined.logits]
    real_maps = [[feature[:batch_size] for feature in group] for group in combined.feature_maps]
    fake_maps = [[feature[batch_size:] for feature in group] for group in combined.feature_maps]
    return (
        DiscriminatorEvaluation(real_logits, real_maps),
        DiscriminatorEvaluation(fake_logits, fake_maps),
    )


def build_discriminator(backend: str) -> VocoderDiscriminator:
    selected = DiscriminatorBackend(backend)
    if selected is DiscriminatorBackend.WAVE_UNET:
        return WaveUNetBackend()
    return StyleTTSBackend()
