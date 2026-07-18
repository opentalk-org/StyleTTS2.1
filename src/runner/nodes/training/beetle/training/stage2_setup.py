import copy
from typing import Protocol

import torch
from torch import nn

from ..config.training import OptimizerConfig, Precision, StageConfig
from ..losses.stage2 import Stage2LossInput
from ..models.model import Stage1Models
from ..models.stage2 import Stage2Models
from .optimizer import (
    NamedGradientGroup,
    OptimizerSet,
    ScheduledOptimizer,
    learning_rate_schedule,
)
from .state import LoopState


class Stage2InputBuilder(Protocol):
    def build(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
    ) -> Stage2LossInput: ...

    def build_validation(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
    ) -> Stage2LossInput: ...


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


def build_stage3_optimizers(
    stage1: Stage1Models,
    stage2: Stage2Models,
    config: StageConfig,
    device: torch.device,
) -> OptimizerSet:
    discriminator_config = config.discriminator_optimizer
    if discriminator_config is None:
        raise ValueError("Stage 3 requires a discriminator optimizer")
    modules = (
        stage1.audio_encoder,
        stage1.feature_linear,
        stage1.decoder,
        stage1.generator,
        *trainable_stage2_modules(stage2),
    )
    parameters = tuple(
        parameter for module in modules for parameter in module.parameters()
    )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("Stage 3 generator parameters must have one optimizer owner")
    scale_enabled = config.precision is Precision.FLOAT16
    generator = _adamw(parameters, config.generator_optimizer)
    discriminator = _adamw(
        tuple(stage1.discriminators.parameters()), discriminator_config
    )
    return OptimizerSet(
        (
            ScheduledOptimizer(
                "discriminator",
                discriminator,
                learning_rate_schedule(discriminator_config),
                torch.amp.GradScaler(device.type, enabled=scale_enabled),
                discriminator_config.maximum_gradient_norm,
            ),
            ScheduledOptimizer(
                "generator",
                generator,
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
        models.language_embedding,
        models.condition_bank,
        models.duration_predictor,
        models.latent_flow,
        models.aligner,
        models.style_speaker_classifier,
        models.style_statistics_head,
        models.voice_ge2e,
        models.style_ge2e,
    )


def named_trainable_stage2_modules(
    models: Stage2Models,
) -> tuple[tuple[str, nn.Module], ...]:
    names = (
        "phoneme_encoder",
        "latent_phoneme_encoder",
        "duration_phoneme_encoder",
        "context_phoneme_encoder",
        "context_audio_encoder",
        "style_encoder",
        "voice_encoder",
        "language_embedding",
        "condition_bank",
        "duration_predictor",
        "latent_flow",
        "aligner",
        "style_speaker_classifier",
        "style_statistics_head",
        "voice_ge2e",
        "style_ge2e",
    )
    return tuple(zip(names, trainable_stage2_modules(models), strict=True))


def stage2_gradient_groups(models: Stage2Models) -> tuple[NamedGradientGroup, ...]:
    return (
        NamedGradientGroup(
            "phoneme_encoders",
            (
                models.phoneme_encoder,
                models.latent_phoneme_encoder,
                models.duration_phoneme_encoder,
            ),
        ),
        NamedGradientGroup(
            "context_encoders",
            (models.context_phoneme_encoder, models.context_audio_encoder),
        ),
        NamedGradientGroup(
            "conditioning",
            (models.language_embedding, models.condition_bank),
        ),
        NamedGradientGroup("style_encoder", (models.style_encoder,)),
        NamedGradientGroup("voice_encoder", (models.voice_encoder,)),
        NamedGradientGroup("duration_predictor", (models.duration_predictor,)),
        NamedGradientGroup("latent_flow", (models.latent_flow,)),
    )


def frozen_stage2_modules(models: Stage2Models) -> tuple[nn.Module, ...]:
    return (
        models.audio_encoder,
        models.f0_extractor,
        models.text_encoder,
    )


@torch.no_grad()
def update_latent_flow_ema(ema: nn.Module, online: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    online_parameters = dict(online.named_parameters())
    if ema_parameters.keys() != online_parameters.keys():
        raise ValueError("EMA and online latent-flow parameters do not match")
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.mul_(decay).add_(online_parameters[name], alpha=1 - decay)
    ema_buffers = dict(ema.named_buffers())
    online_buffers = dict(online.named_buffers())
    if ema_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA and online latent-flow buffers do not match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(online_buffers[name])


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
