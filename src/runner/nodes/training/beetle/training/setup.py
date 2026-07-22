import copy
from typing import Protocol

import torch
from torch import Tensor, nn

from ..config.training import OptimizerConfig, Precision, TrainingConfig
from ..losses.acoustic import masked_f0_smooth_l1, masked_n_smooth_l1
from ..losses.conditional import ConditionalLossInput
from ..models.model import AcousticModels
from ..models.modules.audio import AcousticFeatures
from ..models.conditional import ConditionalModels
from .callbacks import TrainingMetric
from .conditional_features import ConditionalSynthesisInput
from .distributed import DistributedRuntime
from .optimizer import (
    GradientClipping,
    NamedGradientGroup,
    OptimizerSet,
    ScheduledOptimizer,
    learning_rate_schedule,
)
from .state import LoopState


class ConditionalInputBuilder(Protocol):
    def build_synthesis(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
    ) -> ConditionalSynthesisInput: ...

    def build(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
    ) -> ConditionalLossInput: ...

    def build_validation(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
    ) -> ConditionalLossInput: ...


def build_optimizers(
    acoustic: AcousticModels,
    conditional: ConditionalModels,
    config: TrainingConfig,
    runtime: DistributedRuntime,
) -> OptimizerSet:
    discriminator_config = config.discriminator_optimizer
    if discriminator_config is None:
        raise ValueError("training requires a discriminator optimizer")
    modules = (
        acoustic.audio_encoder,
        acoustic.feature_linear,
        acoustic.decoder,
        acoustic.generator,
        *trainable_conditional_modules(conditional),
    )
    parameters = tuple(parameter for module in modules for parameter in module.parameters())
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("generator parameters must have one optimizer owner")
    generator = _adamw(parameters, config.generator_optimizer)
    discriminator = _adamw(
        tuple(acoustic.discriminators.parameters()),
        discriminator_config,
    )
    scale_enabled = config.precision is Precision.FLOAT16
    return OptimizerSet(
        (
            ScheduledOptimizer(
                "discriminator",
                discriminator,
                learning_rate_schedule(discriminator_config),
                torch.amp.GradScaler(
                    runtime.device.type,
                    init_scale=16.0,
                    enabled=scale_enabled,
                ),
                discriminator_config.maximum_gradient_norm,
                runtime,
                (
                    NamedGradientGroup(
                        "discriminators",
                        (acoustic.discriminators,),
                        GradientClipping.CLIP,
                    ),
                ),
            ),
            ScheduledOptimizer(
                "generator",
                generator,
                learning_rate_schedule(config.generator_optimizer),
                torch.amp.GradScaler(
                    runtime.device.type,
                    init_scale=16.0,
                    enabled=scale_enabled,
                ),
                config.generator_optimizer.maximum_gradient_norm,
                runtime,
                (
                    *_acoustic_gradient_groups(acoustic),
                    *_conditional_gradient_groups(conditional),
                ),
            ),
        )
    )


def build_latent_flow_ema(models: ConditionalModels) -> nn.Module:
    return copy.deepcopy(models.latent_flow).requires_grad_(False).eval()


def prepare_training_modules(
    acoustic: AcousticModels,
    conditional: ConditionalModels,
    runtime: DistributedRuntime,
) -> None:
    trainable_acoustic = (
        acoustic.audio_encoder,
        acoustic.feature_linear,
        acoustic.decoder,
        acoustic.generator,
    )
    for module in trainable_acoustic:
        module.to(runtime.device).requires_grad_(True).train()
    acoustic.reconstruction_loss.to(runtime.device).train()
    acoustic.discriminators.to(runtime.device).requires_grad_(True).train()
    acoustic.f0_extractor.to(runtime.device).requires_grad_(False).eval()
    names = ("audio_encoder", "feature_linear", "decoder", "generator", "discriminators")
    for name in names:
        setattr(acoustic, name, runtime.prepare_module(getattr(acoustic, name)))
    conditional.audio_encoder = acoustic.audio_encoder
    conditional.f0_extractor = acoustic.f0_extractor
    conditional.to(runtime.device).train()
    for name, module in named_trainable_conditional_modules(conditional):
        setattr(conditional, name, runtime.prepare_module(module))
    for module in trainable_conditional_modules(conditional):
        module.requires_grad_(True).train()
    conditional.text_encoder.requires_grad_(False).eval()


def mean_acoustic_loss(
    posterior: AcousticFeatures,
    conditional: AcousticFeatures,
    target: Tensor,
    mask: Tensor,
    pitch: bool,
) -> Tensor:
    if pitch:
        loss = masked_f0_smooth_l1
        generated = posterior.f0
        conditioned = conditional.f0
    else:
        loss = masked_n_smooth_l1
        generated = posterior.n
        conditioned = conditional.n
    return 0.5 * (loss(generated, target, mask) + loss(conditioned, target, mask))


def trainable_conditional_modules(models: ConditionalModels) -> tuple[nn.Module, ...]:
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


def named_trainable_conditional_modules(
    models: ConditionalModels,
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
    return tuple(zip(names, trainable_conditional_modules(models), strict=True))


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


def tensor_metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().float()))


def _acoustic_gradient_groups(models: AcousticModels) -> tuple[NamedGradientGroup, ...]:
    return (
        NamedGradientGroup("audio_encoder", (models.audio_encoder,), GradientClipping.CLIP),
        NamedGradientGroup("feature_linear", (models.feature_linear,), GradientClipping.OBSERVE),
        NamedGradientGroup("decoder", (models.decoder,), GradientClipping.CLIP),
        NamedGradientGroup("generator", (models.generator,), GradientClipping.CLIP),
    )


def _conditional_gradient_groups(models: ConditionalModels) -> tuple[NamedGradientGroup, ...]:
    return tuple(
        NamedGradientGroup(name, modules, GradientClipping.CLIP)
        for name, modules in (
            (
                "phoneme_encoders",
                (
                    models.phoneme_encoder,
                    models.latent_phoneme_encoder,
                    models.duration_phoneme_encoder,
                ),
            ),
            ("context_encoders", (models.context_phoneme_encoder, models.context_audio_encoder)),
            ("conditioning", (models.language_embedding, models.condition_bank)),
            ("style_encoder", (models.style_encoder,)),
            ("voice_encoder", (models.voice_encoder,)),
            ("duration_predictor", (models.duration_predictor,)),
            ("latent_flow", (models.latent_flow,)),
            ("aligner", (models.aligner,)),
            (
                "style_auxiliaries",
                (
                    models.style_speaker_classifier,
                    models.style_statistics_head,
                    models.style_ge2e,
                ),
            ),
            ("voice_auxiliaries", (models.voice_ge2e,)),
        )
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
