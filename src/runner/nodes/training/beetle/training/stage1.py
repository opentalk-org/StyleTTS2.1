import torch

from ..config.training import AdversarialConfig, StageConfig
from ..data.prefetch import DataPipelineState
from ..data.sampling import derive_seed
from ..data.stage1_records import Stage1Batch, Stage1WindowGeometry
from ..losses.acoustic import masked_f0_smooth_l1, masked_kl_standard_normal
from ..losses.acoustic import masked_n_smooth_l1
from ..losses.adversarial import discriminator_step_loss, generator_step_loss
from ..losses.composition import Stage1LossWeights
from ..models.model import Stage1Models, Stage1Synthesis
from .callbacks import TrainingMetric
from .checkpoint import CHECKPOINT_VERSION, CheckpointPayload, GradientTarget
from .checkpoint import NamedModuleGradients, StateKind, StateTarget
from .checkpoint import capture_named_state, restore_checkpoint_gradients
from .checkpoint import restore_named_states, validate_resume_fingerprints
from .distributed import DistributedRuntime
from .loop import LoopIntervals
from .optimizer import NamedGradientGroup, OptimizerSet
from .reporting import ReportingState
from .stage1_setup import (
    AlignedSegmentTraining,
    Stage1Schedules,
    build_stage1_optimizers,
    tensor_metric,
)
from .state import LoopState, StageKind, capture_gradients


class Stage1Trainer(AlignedSegmentTraining):
    stage = StageKind.STAGE1
    trains_discriminator = True

    def __init__(
        self,
        models: Stage1Models,
        stage_config: StageConfig,
        adversarial_config: AdversarialConfig,
        runtime_seed: int,
        runtime: DistributedRuntime,
        optimizers: OptimizerSet,
        intervals: LoopIntervals,
        config_fingerprint: str,
        data_fingerprint: str,
        initial_loop: LoopState,
        geometry: Stage1WindowGeometry | None = None,
    ) -> None:
        if initial_loop.stage is not self.stage:
            raise ValueError("Stage 1 trainer requires a Stage 1 loop state")
        self.models = models
        for module in (
            models.audio_encoder, models.feature_linear, models.decoder,
            models.generator, models.reconstruction_loss,
        ):
            module.to(runtime.device).train()
        models.f0_extractor.to(runtime.device).requires_grad_(False).eval()
        models.discriminators.to(runtime.device).requires_grad_(True).train()
        for name in (
            "audio_encoder",
            "feature_linear",
            "decoder",
            "generator",
            "discriminators",
        ):
            setattr(models, name, runtime.prepare_module(getattr(models, name)))
        self.stage_config = stage_config
        self.adversarial_config = adversarial_config
        self.geometry = geometry
        self.runtime_seed = runtime_seed
        self.runtime = runtime
        self.world_size = runtime.world_size
        self.device = runtime.device
        self.optimizers = optimizers.prepare_distributed()
        self.intervals = intervals
        self.config_fingerprint = config_fingerprint
        self.data_fingerprint = data_fingerprint
        self._loop = initial_loop
        self.schedules = Stage1Schedules.from_config(stage_config)
        self.accumulation_steps = stage_config.accumulation_steps

    def loop_state(self) -> LoopState:
        return self._loop

    def set_loop_state(self, state: LoopState) -> None:
        if state.stage is not self.stage:
            raise ValueError("loop stage cannot change during Stage 1 training")
        self._loop = state

    def discriminator_backward(
        self,
        batch: Stage1Batch,
    ) -> tuple[TrainingMetric, ...]:
        values = batch.to(self.device)
        with torch.no_grad(), self.runtime.autocast():
            synthesis = self._synthesize_window(values, "discriminator")
        with self.runtime.autocast():
            loss = discriminator_step_loss(
                self.models.discriminators,
                values.waveform,
                synthesis.waveform,
            )
            weighted = loss * self._weights().discriminator
        scaled = weighted / self.accumulation_steps
        self.optimizers.group("discriminator").backward(scaled)
        return (
            tensor_metric("discriminator", loss),
            tensor_metric("discriminator_total", weighted),
        )

    def generator_backward(
        self,
        batch: Stage1Batch,
    ) -> tuple[TrainingMetric, ...]:
        values = batch.to(self.device)
        f0_target = self.models.f0_target(values.target_mel, values.frame_mask)
        n_target = self.models.n_target(values.target_mel, values.frame_mask)
        with self.runtime.autocast():
            synthesis = self._synthesize_window(values, "generator")
            encoder_kl = masked_kl_standard_normal(
                synthesis.posterior.mean,
                synthesis.posterior.log_scale,
                synthesis.posterior.mask,
            )
            f0 = masked_f0_smooth_l1(
                synthesis.acoustic.f0,
                f0_target,
                values.frame_mask,
            )
            n = masked_n_smooth_l1(
                synthesis.acoustic.n,
                n_target,
                values.frame_mask,
            )
            reconstruction = self.models.reconstruction_loss(
                synthesis.waveform,
                values.waveform,
                synthesis.sample_mask,
            ).total
            adversarial = generator_step_loss(
                self.models.discriminators,
                values.waveform,
                synthesis.waveform,
            )
            weights = self._weights()
            total = (
                encoder_kl * weights.encoder_kl
                + f0 * weights.f0
                + n * weights.n
                + reconstruction * weights.reconstruction
                + adversarial.adversarial * weights.generator_adversarial
                + adversarial.feature_matching * weights.feature_matching
            )
        self.optimizers.group("generator").backward(total / self.accumulation_steps)
        return (
            tensor_metric("encoder_kl", encoder_kl),
            tensor_metric("f0", f0),
            tensor_metric("n", n),
            tensor_metric("reconstruction", reconstruction),
            tensor_metric("generator_adversarial", adversarial.adversarial),
            tensor_metric("feature_matching", adversarial.feature_matching),
            tensor_metric("generator_total", total),
        )

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]:
        return self.optimizers.step(optimizer_step, self.gradient_groups())
    def reduce_metrics(self, metrics: tuple[TrainingMetric, ...]) -> tuple[TrainingMetric, ...]:
        return self.runtime.reduce_metrics(metrics)
    def gradient_groups(self) -> tuple[NamedGradientGroup, ...]:
        return (
            NamedGradientGroup("audio_encoder", (self.models.audio_encoder,)),
            NamedGradientGroup("feature_linear", (self.models.feature_linear,)),
            NamedGradientGroup("decoder", (self.models.decoder,)),
            NamedGradientGroup("generator", (self.models.generator,)),
            NamedGradientGroup("discriminators", (self.models.discriminators,)),
        )

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
        expected_schedule = self.schedules.state(payload.loop.optimizer_step)
        if payload.loss_schedule != expected_schedule:
            raise ValueError("loss schedule state does not match Stage 1 configuration")
        restore_named_states(
            payload.states,
            (*self._model_targets(), *self.optimizers.state_targets()),
        )
        restore_checkpoint_gradients(payload.gradients, self._gradient_targets())
        self._restore_rank_state(payload)
        self._loop = payload.loop
        return payload.sampler_state

    def _restore_rank_state(self, payload: CheckpointPayload) -> None:
        if len(payload.rank_states) != self.runtime.world_size:
            raise ValueError("checkpoint world size does not match distributed runtime")
        self.runtime.restore_rank_state(payload.rank_states[self.runtime.rank])

    def _synthesize_window(
        self,
        batch: Stage1Batch,
        view: str,
    ) -> Stage1Synthesis:
        if self.geometry is None:
            raise RuntimeError("Stage 1 window geometry is required")
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
        return self.models.reconstruct_window(
            batch.encoder_mel,
            batch.encoder_mask,
            batch.frame_mask,
            self.geometry.posterior_start,
            self.geometry.latent_frames,
            latent,
            source,
        )

    def _weights(self) -> Stage1LossWeights:
        return self.schedules.weights(self._loop.optimizer_step)

    def _model_states(self):
        return tuple(
            capture_named_state(name, kind, module) for name, kind, module in self._state_modules()
        )

    def _model_targets(self):
        return tuple(
            StateTarget(name, kind, module) for name, kind, module in self._state_modules()
        )

    def _state_modules(self):
        return (
            ("audio_encoder", StateKind.MODEL, self.runtime.unwrap(self.models.audio_encoder)),
            ("feature_linear", StateKind.MODEL, self.runtime.unwrap(self.models.feature_linear)),
            ("decoder", StateKind.MODEL, self.runtime.unwrap(self.models.decoder)),
            ("generator", StateKind.MODEL, self.runtime.unwrap(self.models.generator)),
            ("f0_extractor", StateKind.FROZEN_MODEL, self.models.f0_extractor),
            ("discriminators", StateKind.DISCRIMINATOR, self.runtime.unwrap(self.models.discriminators)),
        )

    def _gradients(self) -> tuple[NamedModuleGradients, ...]:
        return tuple(
            NamedModuleGradients(name, capture_gradients(module))
            for name, _, module in self._state_modules()
            if name != "f0_extractor"
        )

    def _gradient_targets(self) -> tuple[GradientTarget, ...]:
        return tuple(
            GradientTarget(name, module)
            for name, _, module in self._state_modules()
            if name != "f0_extractor"
        )
