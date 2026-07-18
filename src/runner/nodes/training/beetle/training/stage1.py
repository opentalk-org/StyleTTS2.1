from contextlib import AbstractContextManager

import torch
from torch import Tensor

from ..config.training import AdversarialConfig, Precision, StageConfig
from ..data.prefetch import DataPipelineState
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.acoustic import masked_f0_smooth_l1, masked_kl_standard_normal
from ..losses.acoustic import masked_n_smooth_l1
from ..losses.adversarial import discriminator_step_loss, generator_step_loss
from ..losses.composition import Stage1LossWeights
from ..models.model import Stage1Models, Stage1Synthesis
from ..models.modules.segments import AlignedSegments
from .callbacks import TrainingMetric
from .checkpoint import CHECKPOINT_VERSION, CheckpointPayload, GradientTarget
from .checkpoint import NamedModuleGradients, StateKind, StateTarget
from .checkpoint import capture_named_state, restore_checkpoint_gradients
from .checkpoint import restore_named_states, validate_resume_fingerprints
from .loop import LoopIntervals
from .optimizer import NamedGradientGroup, OptimizerSet
from .reporting import ReportingState
from .stage1_setup import Stage1Schedules, build_stage1_optimizers, tensor_metric
from .state import LoopState, StageKind, capture_gradients, capture_rng_state
from .state import restore_rng_state


class Stage1Trainer:
    stage = StageKind.STAGE1
    trains_discriminator = True

    def __init__(
        self,
        models: Stage1Models,
        stage_config: StageConfig,
        adversarial_config: AdversarialConfig,
        runtime_seed: int,
        device: torch.device,
        optimizers: OptimizerSet,
        intervals: LoopIntervals,
        config_fingerprint: str,
        data_fingerprint: str,
        initial_loop: LoopState,
    ) -> None:
        if initial_loop.stage is not self.stage:
            raise ValueError("Stage 1 trainer requires a Stage 1 loop state")
        self.models = models
        for module in (
            models.audio_encoder, models.feature_linear, models.decoder,
            models.generator, models.reconstruction_loss,
        ):
            module.to(device).train()
        models.f0_extractor.to(device).requires_grad_(False).eval()
        models.discriminators.to(device).requires_grad_(True).train()
        self.stage_config = stage_config
        self.adversarial_config = adversarial_config
        self.runtime_seed = runtime_seed
        self.device = device
        self.optimizers = optimizers
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
        batch: BeetleBatch,
    ) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        segment = self._segment(frame_mask, "discriminator")
        real = segment.samples(waveform)
        with torch.no_grad(), self._autocast():
            synthesis = self._synthesize(mel, frame_mask, segment, "discriminator")
        with self._autocast():
            loss = discriminator_step_loss(
                self.models.discriminators, real, synthesis.waveform
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
        batch: BeetleBatch,
    ) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        segment = self._segment(frame_mask, "generator")
        real = segment.samples(waveform)
        f0_target = self.models.segment_f0_target(mel, frame_mask, segment)
        n_target = self.models.n_target(mel, frame_mask)
        segment_mask = segment.frames(frame_mask)
        with self._autocast():
            synthesis = self._synthesize(mel, frame_mask, segment, "generator")
            encoder_kl = masked_kl_standard_normal(
                synthesis.posterior.mean,
                synthesis.posterior.log_scale,
                synthesis.posterior.mask,
            )
            f0 = masked_f0_smooth_l1(
                segment.frames(synthesis.acoustic.f0), f0_target, segment_mask
            )
            n = masked_n_smooth_l1(synthesis.acoustic.n, n_target, frame_mask)
            reconstruction = self.models.reconstruction_loss(
                synthesis.waveform,
                real,
                synthesis.sample_mask,
            ).total
            adversarial = generator_step_loss(
                self.models.discriminators, real, synthesis.waveform
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
            rng=capture_rng_state(),
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
        restore_rng_state(payload.rng)
        self._loop = payload.loop
        return payload.sampler_state

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
        return self.models.reconstruct_segment(mel, frame_mask, segment, latent, source)

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
        return AlignedSegments.random(
            frame_mask,
            self.adversarial_config.segment_samples
            // self.models.generator.config.output_hop(),
            self.models.audio_encoder.config.downsample_rate,
            self.models.generator.config.output_hop(),
            generator,
        )

    def _inputs(self, batch: BeetleBatch) -> tuple[Tensor, Tensor, Tensor]:
        return (
            batch.waveform.to(self.device, non_blocking=True),
            batch.mel.to(self.device, non_blocking=True),
            batch.frame_mask.to(self.device, non_blocking=True),
        )

    def _autocast(self) -> AbstractContextManager[None]:
        if self.stage_config.precision is Precision.FLOAT32:
            return torch.autocast(self.device.type, enabled=False)
        dtype = (
            torch.bfloat16
            if self.stage_config.precision is Precision.BFLOAT16
            else torch.float16
        )
        return torch.autocast(self.device.type, dtype=dtype)

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
            ("audio_encoder", StateKind.MODEL, self.models.audio_encoder),
            ("feature_linear", StateKind.MODEL, self.models.feature_linear),
            ("decoder", StateKind.MODEL, self.models.decoder),
            ("generator", StateKind.MODEL, self.models.generator),
            ("f0_extractor", StateKind.FROZEN_MODEL, self.models.f0_extractor),
            ("discriminators", StateKind.DISCRIMINATOR, self.models.discriminators),
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
