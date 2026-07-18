from dataclasses import dataclass

from ..config.training import StageConfig
from ..losses.composition import Stage1LossWeights
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
    stage1: Stage1Schedules
    stage2: Stage2Schedules

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage3Schedules":
        return cls(
            Stage1Schedules.from_config(config),
            Stage2Schedules.from_config(config),
        )

    def weights(self, optimizer_step: int) -> Stage1LossWeights:
        return self.stage1.weights(optimizer_step)

    def state(self, optimizer_step: int) -> LossScheduleState:
        stage1 = self.stage1.state(optimizer_step)
        stage2 = self.stage2.state(optimizer_step)
        return LossScheduleState(
            optimizer_step,
            (*stage1.weights, *stage2.weights),
        )
