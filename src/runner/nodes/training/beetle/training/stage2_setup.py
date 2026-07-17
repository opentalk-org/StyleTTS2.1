import copy
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ..config.training import OptimizerConfig, Precision, StageConfig
from ..losses.stage2 import Stage2LossInput, Stage2LossWeights
from ..models.stage2 import Stage2Models
from .callbacks import TrainingCallbacks, TrainingMetric
from .checkpoint import LossScheduleState, LossWeight
from .optimizer import (
    OptimizerSet,
    ScheduledOptimizer,
    StepSchedule,
    learning_rate_schedule,
    loss_weight_schedule,
)
from .state import LoopState

_LOSS_NAMES = (
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


class Stage2InputBuilder(Protocol):
    def build(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
    ) -> Stage2LossInput: ...


class Stage2Validator(Protocol):
    def run(
        self,
        optimizer_step: int,
        callbacks: TrainingCallbacks,
    ) -> tuple[TrainingMetric, ...]: ...


@dataclass(frozen=True)
class Stage2Schedules:
    values: tuple[StepSchedule, ...]

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage2Schedules":
        losses = config.losses
        scheduled = tuple(
            loss_weight_schedule(getattr(losses, name)) for name in _LOSS_NAMES
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
                for name, value in zip(_LOSS_NAMES, weights.values(), strict=True)
            ),
        )


def build_stage2_optimizer(
    models: Stage2Models,
    config: StageConfig,
    device: torch.device,
) -> OptimizerSet:
    if config.discriminator_optimizer is not None:
        raise ValueError("Stage 2 must not configure a discriminator optimizer")
    parameters = tuple(
        parameter
        for module in trainable_stage2_modules(models)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    optimizer = _adamw(parameters, config.generator_optimizer)
    scale_enabled = config.precision is Precision.FLOAT16
    return OptimizerSet(
        (
            ScheduledOptimizer(
                "generator",
                optimizer,
                learning_rate_schedule(config.generator_optimizer),
                torch.amp.GradScaler(device.type, enabled=scale_enabled),
                config.generator_optimizer.maximum_gradient_norm,
            ),
        )
    )


def build_latent_flow_ema(models: Stage2Models) -> nn.Module:
    return copy.deepcopy(models.latent_flow).requires_grad_(False).eval()


def trainable_stage2_modules(models: Stage2Models) -> tuple[nn.Module, ...]:
    return (
        models.phoneme_encoder,
        models.latent_phoneme_encoder,
        models.duration_phoneme_encoder,
        models.context_phoneme_encoder,
        models.context_audio_encoder,
        models.style_encoder,
        models.voice_encoder,
        models.condition_bank,
        models.duration_predictor,
        models.latent_flow,
        models.style_speaker_classifier,
        models.style_statistics_head,
        models.voice_ge2e,
        models.style_ge2e,
    )


def frozen_stage2_modules(models: Stage2Models) -> tuple[nn.Module, ...]:
    return (
        models.audio_encoder,
        models.f0_extractor,
        models.aligner,
        models.text_encoder,
    )


def _adamw(
    parameters: tuple[nn.Parameter, ...],
    config: OptimizerConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
