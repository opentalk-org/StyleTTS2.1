from dataclasses import dataclass
import torch
from torch import Tensor, nn

from ..config.training import (
    AdversarialConfig,
    OptimizerConfig,
    Precision,
    Stage1ConditioningConfig,
    StageConfig,
)
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.composition import Stage1LossWeights
from ..models.model import Stage1Models, Stage1Synthesis
from ..models.modules.segments import AlignedSegments
from .callbacks import TrainingMetric
from .checkpoint import LossScheduleState, LossWeight
from .distributed import DistributedRuntime
from .optimizer import (
    GradientClipping,
    NamedGradientGroup,
    OptimizerSet,
    ScheduledOptimizer,
    StepSchedule,
    learning_rate_schedule,
    loss_weight_schedule,
)
from .state import LoopState, StageKind


class AlignedSegmentTraining:
    models: Stage1Models
    adversarial_config: AdversarialConfig
    runtime_seed: int
    stage: StageKind
    device: torch.device
    _loop: LoopState

    def _synthesize(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        segment: AlignedSegments,
        view: str,
    ) -> Stage1Synthesis:
        state = self._loop
        latent_seed = derive_seed(
            self.runtime_seed,
            self.stage,
            state.cycle,
            state.batch_index,
            view,
            "latent",
        )
        source_seed = derive_seed(
            self.runtime_seed,
            self.stage,
            state.cycle,
            state.batch_index,
            view,
            "source",
        )
        latent = torch.Generator(device=self.device).manual_seed(latent_seed)
        source = torch.Generator(device=self.device).manual_seed(source_seed)
        return self.models.reconstruct_segment(
            mel,
            frame_mask,
            segment,
            latent,
            source,
        )

    def _segment(self, frame_mask: Tensor, view: str) -> AlignedSegments:
        state = self._loop
        seed = derive_seed(
            self.runtime_seed,
            self.stage,
            state.cycle,
            state.batch_index,
            view,
            "segment",
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        frame_count = self.adversarial_config.segment_samples // self.models.output_hop
        lengths = frame_mask[:, 0].sum(dim=1).clamp_min(frame_count)
        positions = torch.arange(frame_mask.shape[-1], device=frame_mask.device)
        segment_mask = positions.view(1, 1, -1) < lengths.view(-1, 1, 1)
        return AlignedSegments.random(
            segment_mask,
            frame_count,
            self.models.latent_downsample_rate,
            self.models.output_hop,
            generator,
        )

    def _inputs(self, batch: BeetleBatch) -> tuple[Tensor, Tensor, Tensor]:
        return (
            batch.waveform.to(self.device, non_blocking=True),
            batch.mel.to(self.device, non_blocking=True),
            batch.frame_mask.to(self.device, non_blocking=True),
        )


@dataclass(frozen=True)
class Stage1Schedules:
    encoder_kl: StepSchedule
    f0: StepSchedule
    n: StepSchedule
    reconstruction: StepSchedule
    discriminator: StepSchedule
    generator_adversarial: StepSchedule
    feature_matching: StepSchedule

    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage1Schedules":
        losses = config.losses
        return cls(
            loss_weight_schedule(losses.encoder_kl),
            loss_weight_schedule(losses.f0),
            loss_weight_schedule(losses.n),
            loss_weight_schedule(losses.reconstruction),
            loss_weight_schedule(losses.discriminator),
            loss_weight_schedule(losses.generator_adversarial),
            loss_weight_schedule(losses.feature_matching),
        )

    def weights(self, optimizer_step: int) -> Stage1LossWeights:
        return Stage1LossWeights(
            encoder_kl=self.encoder_kl.value(optimizer_step),
            f0=self.f0.value(optimizer_step),
            n=self.n.value(optimizer_step),
            reconstruction=self.reconstruction.value(optimizer_step),
            discriminator=self.discriminator.value(optimizer_step),
            generator_adversarial=self.generator_adversarial.value(optimizer_step),
            feature_matching=self.feature_matching.value(optimizer_step),
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
                LossWeight("discriminator", weights.discriminator),
                LossWeight("generator_adversarial", weights.generator_adversarial),
                LossWeight("feature_matching", weights.feature_matching),
            ),
        )


@dataclass(frozen=True)
class Stage1ConditioningSchedule:
    predicted_start_step: int
    transition_steps: int

    @classmethod
    def from_config(
        cls,
        config: Stage1ConditioningConfig,
    ) -> "Stage1ConditioningSchedule":
        return cls(config.predicted_start_step, config.transition_steps)

    def predicted_ratio(self, optimizer_step: int) -> float:
        if self.predicted_start_step == 0:
            return 1.0
        transition_position = optimizer_step - self.predicted_start_step
        bounded = min(max(transition_position, 0), self.transition_steps)
        return bounded / self.transition_steps


def build_stage1_optimizers(
    models: Stage1Models,
    config: StageConfig,
    runtime: DistributedRuntime,
) -> OptimizerSet:
    discriminator_config = config.discriminator_optimizer
    if discriminator_config is None:
        raise ValueError("Stage 1 requires a discriminator optimizer")
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
    discriminator = _adamw(
        tuple(models.discriminators.parameters()), discriminator_config
    )
    scale_enabled = config.precision is Precision.FLOAT16
    return OptimizerSet(
        (
            ScheduledOptimizer(
                "discriminator",
                discriminator,
                learning_rate_schedule(discriminator_config),
                torch.amp.GradScaler(runtime.device.type, enabled=scale_enabled),
                discriminator_config.maximum_gradient_norm,
                runtime,
                stage1_discriminator_gradient_groups(models),
            ),
            ScheduledOptimizer(
                "generator",
                generator,
                learning_rate_schedule(config.generator_optimizer),
                torch.amp.GradScaler(runtime.device.type, enabled=scale_enabled),
                config.generator_optimizer.maximum_gradient_norm,
                runtime,
                stage1_generator_gradient_groups(models),
            ),
        )
    )


def stage1_generator_gradient_groups(
    models: Stage1Models,
) -> tuple[NamedGradientGroup, ...]:
    return (
        NamedGradientGroup(
            "audio_encoder", (models.audio_encoder,), GradientClipping.CLIP
        ),
        NamedGradientGroup(
            "feature_linear", (models.feature_linear,), GradientClipping.OBSERVE
        ),
        NamedGradientGroup("decoder", (models.decoder,), GradientClipping.CLIP),
        NamedGradientGroup("generator", (models.generator,), GradientClipping.CLIP),
    )


def stage1_discriminator_gradient_groups(
    models: Stage1Models,
) -> tuple[NamedGradientGroup, ...]:
    return (
        NamedGradientGroup(
            "discriminators", (models.discriminators,), GradientClipping.CLIP
        ),
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
