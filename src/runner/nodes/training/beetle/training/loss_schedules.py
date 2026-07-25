import math
from dataclasses import dataclass
from typing import Any

from ..config.training import OptimizerConfig, ScheduledWeight, TrainingConfig
from ..losses.composition import AcousticLossWeights
from ..losses.conditional import ConditionalLossWeights
from .checkpoint import LossScheduleState, LossWeight

_ACOUSTIC_NAMES = (
    "encoder_kl",
    "f0",
    "n",
    "reconstruction",
    "discriminator",
    "generator_adversarial",
    "feature_matching",
)
_CONDITIONAL_NAMES = (
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
class StepSchedule:
    start_step: int
    warmup_steps: int
    decay_steps: int
    initial_value: float
    peak_value: float
    final_value: float

    def value(self, optimizer_step: int) -> float:
        relative = optimizer_step - self.start_step
        if relative < 0:
            return self.initial_value
        if self.warmup_steps > 0 and relative < self.warmup_steps:
            ratio = relative / self.warmup_steps
            return self.initial_value + ratio * (self.peak_value - self.initial_value)
        decay_position = relative - self.warmup_steps
        if self.decay_steps == 0 or decay_position >= self.decay_steps:
            return self.final_value
        ratio = decay_position / self.decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.final_value + cosine * (self.peak_value - self.final_value)

    @classmethod
    def loss_weight(
        cls,
        value: float,
        start_step: int,
        warmup_steps: int,
    ) -> "StepSchedule":
        return cls(start_step, warmup_steps, 0, 0.0, value, value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "initial_value": self.initial_value,
            "peak_value": self.peak_value,
            "final_value": self.final_value,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
@dataclass(frozen=True)
class TrainingSchedules:
    acoustic: tuple[StepSchedule, ...]
    conditional: tuple[StepSchedule, ...]
    acoustic_prediction: StepSchedule

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingSchedules":
        losses = config.losses
        return cls(
            tuple(loss_weight_schedule(getattr(losses, name)) for name in _ACOUSTIC_NAMES),
            tuple(
                loss_weight_schedule(getattr(losses, name))
                for name in _CONDITIONAL_NAMES
            ),
            loss_weight_schedule(config.acoustic_prediction),
        )

    def acoustic_weights(self, optimizer_step: int) -> AcousticLossWeights:
        values = tuple(schedule.value(optimizer_step) for schedule in self.acoustic)
        return AcousticLossWeights(*values)

    def conditional_weights(self, optimizer_step: int) -> ConditionalLossWeights:
        values = tuple(schedule.value(optimizer_step) for schedule in self.conditional)
        return ConditionalLossWeights(*values)

    def predicted_acoustic_ratio(self, optimizer_step: int) -> float:
        return self.acoustic_prediction.value(optimizer_step)

    def state(self, optimizer_step: int) -> LossScheduleState:
        values = (
            *(schedule.value(optimizer_step) for schedule in self.acoustic),
            *(schedule.value(optimizer_step) for schedule in self.conditional),
        )
        return LossScheduleState(
            optimizer_step,
            tuple(
                LossWeight(name, value)
                for name, value in zip(
                    (*_ACOUSTIC_NAMES, *_CONDITIONAL_NAMES),
                    values,
                    strict=True,
                )
            ),
        )

    @property
    def conditional_names(self) -> tuple[str, ...]:
        return _CONDITIONAL_NAMES


def learning_rate_schedule(config: OptimizerConfig) -> StepSchedule:
    return StepSchedule(
        start_step=0,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        initial_value=0.0,
        peak_value=config.learning_rate,
        final_value=config.learning_rate * config.minimum_learning_rate_ratio,
    )


def loss_weight_schedule(config: ScheduledWeight) -> StepSchedule:
    return StepSchedule.loss_weight(
        value=config.value,
        start_step=config.start_step,
        warmup_steps=config.warmup_steps,
    )
