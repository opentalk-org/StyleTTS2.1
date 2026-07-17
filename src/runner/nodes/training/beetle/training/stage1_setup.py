from dataclasses import dataclass
import torch
from torch import Tensor, nn

from ..config.training import OptimizerConfig, Precision, StageConfig
from ..models.model import Stage1Models
from .callbacks import TrainingMetric
from .checkpoint import LossScheduleState, LossWeight
from .optimizer import (
    OptimizerSet,
    ScheduledOptimizer,
    StepSchedule,
    learning_rate_schedule,
    loss_weight_schedule,
)


@dataclass(frozen=True)
class AcousticLossWeights:
    encoder_kl: float
    f0: float
    n: float
    reconstruction: float


@dataclass(frozen=True)
class Stage1Schedules:
    encoder_kl: StepSchedule
    f0: StepSchedule
    n: StepSchedule
    reconstruction: StepSchedule

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage1Schedules":
        losses = config.losses
        return cls(
            loss_weight_schedule(losses.encoder_kl),
            loss_weight_schedule(losses.f0),
            loss_weight_schedule(losses.n),
            loss_weight_schedule(losses.reconstruction),
        )

    def weights(self, optimizer_step: int) -> AcousticLossWeights:
        return AcousticLossWeights(
            encoder_kl=self.encoder_kl.value(optimizer_step),
            f0=self.f0.value(optimizer_step),
            n=self.n.value(optimizer_step),
            reconstruction=self.reconstruction.value(optimizer_step),
        )

    def state(self, optimizer_step: int) -> LossScheduleState:
        weights = self.weights(optimizer_step)
        return LossScheduleState(
            optimizer_step,
            (
                LossWeight("encoder_kl", weights.encoder_kl),
                LossWeight("f0", weights.f0),
                LossWeight("n", weights.n),
                LossWeight("reconstruction", weights.reconstruction),
            ),
        )


def build_stage1_optimizers(
    models: Stage1Models,
    config: StageConfig,
    device: torch.device,
) -> OptimizerSet:
    if config.discriminator_optimizer is not None:
        raise ValueError("Stage 1 must not configure a discriminator optimizer")
    generator_parameters = tuple(
        parameter
        for module in (
            models.audio_encoder,
            models.feature_linear,
            models.decoder,
            models.generator,
        )
        for parameter in module.parameters()
    )
    generator = _adamw(generator_parameters, config.generator_optimizer)
    scale_enabled = config.precision is Precision.FLOAT16
    return OptimizerSet(
        (
            ScheduledOptimizer(
                "generator",
                generator,
                learning_rate_schedule(config.generator_optimizer),
                torch.amp.GradScaler(device.type, enabled=scale_enabled),
                config.generator_optimizer.maximum_gradient_norm,
            ),
        )
    )


def tensor_metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().float()))


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
