from torch import nn

from ..config.training import StageConfig
from ..data.prefetch import DataPipelineState
from ..data.records import BeetleBatch
from ..losses.stage2 import compute_stage2_losses
from ..models.stage2 import Stage2Models
from .callbacks import TrainingMetric
from .checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointPayload,
    GradientTarget,
    NamedModuleGradients,
    StateKind,
    StateTarget,
    capture_named_state,
    restore_checkpoint_gradients,
    restore_named_states,
    validate_resume_fingerprints,
)
from .distributed import DistributedRuntime
from .loop import LoopIntervals
from .loss_schedules import Stage2Schedules
from .optimizer import NamedGradientGroup, OptimizerSet
from .reporting import ReportingState
from .stage1_setup import tensor_metric
from .stage2_setup import (
    Stage2InputBuilder,
    build_latent_flow_ema,
    build_stage2_optimizer,
    frozen_stage2_modules,
    named_trainable_stage2_modules,
    stage2_gradient_groups,
    update_latent_flow_ema,
)
from .state import (
    LoopState,
    StageKind,
    capture_gradients,
)

__all__ = [
    "Stage2InputBuilder",
    "Stage2Trainer",
    "build_latent_flow_ema",
    "build_stage2_optimizer",
]


class Stage2Trainer:
    stage = StageKind.STAGE2
    trains_discriminator = False

    def __init__(
        self,
        models: Stage2Models,
        ema_latent_flow: nn.Module,
        stage_config: StageConfig,
        runtime: DistributedRuntime,
        optimizers: OptimizerSet,
        intervals: LoopIntervals,
        config_fingerprint: str,
        data_fingerprint: str,
        initial_loop: LoopState,
        input_builder: Stage2InputBuilder,
    ) -> None:
        if initial_loop.stage is not self.stage:
            raise ValueError("Stage 2 trainer requires a Stage 2 loop state")
        self.models = models.to(runtime.device).train()
        for module in frozen_stage2_modules(self.models):
            module.requires_grad_(False).eval()
        for name, module in named_trainable_stage2_modules(self.models):
            setattr(self.models, name, runtime.prepare_module(module))
        self.ema_latent_flow = ema_latent_flow.to(runtime.device).requires_grad_(False).eval()
        self.stage_config = stage_config
        self.runtime = runtime
        self.world_size = runtime.world_size
        self.device = runtime.device
        self.optimizers = optimizers.prepare_distributed()
        self.intervals = intervals
        self.config_fingerprint = config_fingerprint
        self.data_fingerprint = data_fingerprint
        self._loop = initial_loop
        self.input_builder = input_builder
        self.schedules = Stage2Schedules.from_config(stage_config)
        self.accumulation_steps = stage_config.accumulation_steps

    def loop_state(self) -> LoopState:
        return self._loop

    def set_loop_state(self, state: LoopState) -> None:
        if state.stage is not self.stage:
            raise ValueError("loop stage cannot change during Stage 2 training")
        self._loop = state

    def discriminator_backward(
        self,
        batch: BeetleBatch,
    ) -> tuple[TrainingMetric, ...]:
        del batch
        raise RuntimeError("Stage 2 has no discriminator pass")

    def generator_backward(
        self,
        batch: BeetleBatch,
    ) -> tuple[TrainingMetric, ...]:
        inputs = self.input_builder.build(self.models, batch, self._loop)
        with self.runtime.autocast():
            losses = compute_stage2_losses(
                self.models,
                self.ema_latent_flow,
                inputs,
            )
            weights = self.schedules.weights(self._loop.optimizer_step)
            total = losses.total(weights)
        self.optimizers.group("generator").backward(total / self.accumulation_steps)
        names = tuple(weight.name for weight in self.schedules.state(0).weights)
        return (
            *tuple(
                tensor_metric(name, value)
                for name, value in zip(names, losses.values(), strict=True)
            ),
            tensor_metric("stage2_total", total),
        )

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]:
        metrics = self.optimizers.step(optimizer_step, self.gradient_groups())
        update_latent_flow_ema(
            self.ema_latent_flow,
            self.runtime.unwrap(self.models.latent_flow),
            self.runtime.unwrap(self.models.latent_flow).config.ema_decay,
        )
        return metrics

    def reduce_metrics(
        self, metrics: tuple[TrainingMetric, ...]
    ) -> tuple[TrainingMetric, ...]:
        return self.runtime.reduce_metrics(metrics)

    def gradient_groups(self) -> tuple[NamedGradientGroup, ...]:
        return stage2_gradient_groups(self.models)

    def checkpoint_payload(
        self,
        loop: LoopState,
        sampler_state: DataPipelineState,
        reporting: ReportingState,
    ) -> CheckpointPayload:
        return CheckpointPayload(
            version=CHECKPOINT_VERSION,
            config_fingerprint=self.config_fingerprint,
            data_fingerprint=self.data_fingerprint,
            loop=loop,
            rank_states=(self.runtime.capture_rank_state(),),
            states=(*self._model_states(), *self.optimizers.capture_states()),
            gradients=self._gradients(),
            sampler_state=sampler_state,
            loss_schedule=self.schedules.state(loop.optimizer_step),
            reporting=reporting,
        )

    def restore(self, payload: CheckpointPayload) -> DataPipelineState:
        validate_resume_fingerprints(
            payload,
            self.stage,
            self.config_fingerprint,
            self.data_fingerprint,
        )
        expected = self.schedules.state(payload.loop.optimizer_step)
        if payload.loss_schedule != expected:
            raise ValueError("loss schedule state does not match Stage 2 configuration")
        restore_named_states(
            payload.states,
            (*self._model_targets(), *self.optimizers.state_targets()),
        )
        restore_checkpoint_gradients(payload.gradients, self._gradient_targets())
        if len(payload.rank_states) != self.runtime.world_size:
            raise ValueError("checkpoint world size does not match distributed runtime")
        self.runtime.restore_rank_state(payload.rank_states[self.runtime.rank])
        self._loop = payload.loop
        return payload.sampler_state

    def _state_modules(self) -> tuple[tuple[str, StateKind, nn.Module], ...]:
        trainable = tuple(
            (name, StateKind.MODEL, module)
            for name, module in self._named_trainable_modules()
        )
        frozen = tuple(
            (name, StateKind.FROZEN_MODEL, module)
            for name, module in _named_frozen_modules(self.models)
        )
        ema = (("latent_flow", StateKind.EMA, self.ema_latent_flow),)
        return (*trainable, *frozen, *ema)

    def _model_states(self):
        return tuple(
            capture_named_state(name, kind, module)
            for name, kind, module in self._state_modules()
        )

    def _model_targets(self):
        return tuple(
            StateTarget(name, kind, module)
            for name, kind, module in self._state_modules()
        )

    def _gradients(self) -> tuple[NamedModuleGradients, ...]:
        return tuple(
            NamedModuleGradients(name, capture_gradients(module))
            for name, module in self._named_trainable_modules()
        )

    def _gradient_targets(self) -> tuple[GradientTarget, ...]:
        return tuple(
            GradientTarget(name, module)
            for name, module in self._named_trainable_modules()
        )

    def _named_trainable_modules(self) -> tuple[tuple[str, nn.Module], ...]:
        return tuple(
            (name, self.runtime.unwrap(module))
            for name, module in named_trainable_stage2_modules(self.models)
        )


def _named_frozen_modules(models: Stage2Models) -> tuple[tuple[str, nn.Module], ...]:
    names = ("audio_encoder", "f0_extractor", "text_encoder")
    return tuple(zip(names, frozen_stage2_modules(models), strict=True))
