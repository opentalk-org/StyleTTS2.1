from dataclasses import dataclass

from ..config.training import StageConfig
from ..losses.stage2 import Stage2LossWeights
from .checkpoint import LossScheduleState, LossWeight
from .optimizer import StepSchedule, loss_weight_schedule
from .stage1_setup import Stage1Schedules

_STAGE2_LOSS_NAMES = (
    "duration_flow",
    "latent_flow",
    "shortcut",
    "align_s2s",
    "align_mono",
    "align_ctc",
    "voice_contrastive",
    "voice_ge2e",
    "style_contrastive",
    "style_ge2e",
    "style_speaker_adversarial",
    "style_statistics",
    "style_reencoding",
)


@dataclass(frozen=True)
class Stage3LossWeights:
    encoder_kl: float
    f0: float
    n: float
    reconstruction: float
    discriminator: float
    generator_adversarial: float
    feature_matching: float


@dataclass(frozen=True)
class AdversarialSchedules:
    discriminator: StepSchedule
    generator_adversarial: StepSchedule
    feature_matching: StepSchedule

    @classmethod
    def from_config(cls, config: StageConfig) -> "AdversarialSchedules":
        losses = config.losses
        return cls(
            loss_weight_schedule(losses.discriminator),
            loss_weight_schedule(losses.generator_adversarial),
            loss_weight_schedule(losses.feature_matching),
        )

    def state(self, optimizer_step: int) -> LossScheduleState:
        values = (
            self.discriminator.value(optimizer_step),
            self.generator_adversarial.value(optimizer_step),
            self.feature_matching.value(optimizer_step),
        )
        names = ("discriminator", "generator_adversarial", "feature_matching")
        return LossScheduleState(
            optimizer_step,
            tuple(
                LossWeight(name, value)
                for name, value in zip(names, values, strict=True)
            ),
        )


@dataclass(frozen=True)
class Stage2Schedules:
    values: tuple[StepSchedule, ...]

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage2Schedules":
        losses = config.losses
        scheduled = tuple(
            loss_weight_schedule(getattr(losses, name))
            for name in _STAGE2_LOSS_NAMES
        )
        return cls(scheduled)

    def weights(self, optimizer_step: int) -> Stage2LossWeights:
        values = tuple(schedule.value(optimizer_step) for schedule in self.values)
        return Stage2LossWeights(*values)

    def state(self, optimizer_step: int) -> LossScheduleState:
        weights = self.weights(optimizer_step)
        return LossScheduleState(
            optimizer_step,
            tuple(
                LossWeight(name, value)
                for name, value in zip(
                    _STAGE2_LOSS_NAMES,
                    weights.values(),
                    strict=True,
                )
            ),
        )


@dataclass(frozen=True)
class Stage3Schedules:
    acoustic: Stage1Schedules
    adversarial: AdversarialSchedules
    stage2: Stage2Schedules

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage3Schedules":
        return cls(
            Stage1Schedules.from_config(config),
            AdversarialSchedules.from_config(config),
            Stage2Schedules.from_config(config),
        )

    def weights(self, optimizer_step: int) -> Stage3LossWeights:
        acoustic = self.acoustic.weights(optimizer_step)
        return Stage3LossWeights(
            acoustic.encoder_kl,
            acoustic.f0,
            acoustic.n,
            acoustic.reconstruction,
            self.adversarial.discriminator.value(optimizer_step),
            self.adversarial.generator_adversarial.value(optimizer_step),
            self.adversarial.feature_matching.value(optimizer_step),
        )

    def state(self, optimizer_step: int) -> LossScheduleState:
        acoustic = self.acoustic.state(optimizer_step)
        adversarial = self.adversarial.state(optimizer_step)
        stage2 = self.stage2.state(optimizer_step)
        return LossScheduleState(
            optimizer_step,
            (*acoustic.weights, *adversarial.weights, *stage2.weights),
        )
