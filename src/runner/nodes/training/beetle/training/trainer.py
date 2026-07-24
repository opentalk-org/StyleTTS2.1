import torch
from torch import Tensor, nn

from ..config.training import AdversarialConfig, TrainingConfig
from ..data.prefetch import DataPipelineState
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.acoustic import (
    masked_f0_smooth_l1,
    masked_kl_standard_normal,
    masked_n_smooth_l1,
)
from ..losses.adversarial import discriminator_step_loss, generator_step_loss
from ..losses.conditional import compute_conditional_losses
from ..models.model import AcousticModels, AcousticSynthesis
from ..models.modules.audio import AcousticFeatures
from ..models.modules.segments import AlignedSegments
from ..models.conditional import ConditionalModels
from .callbacks import TrainingMetric
from .checkpoint import CheckpointPayload
from .acoustic_synthesis import AcousticBackwardMetrics, synthesize_training_posterior
from .diagnostics import diagnostics_due
from .distributed import DistributedRuntime
from .loop import LoopIntervals
from .loss_schedules import TrainingSchedules
from .optimizer import OptimizerSet
from .reporting import ReportingState
from .setup import (
    ConditionalInputBuilder,
    prepare_training_modules,
    tensor_metric,
    update_latent_flow_ema,
)
from .state import LoopState
from .trainer_checkpoint import checkpoint_payload, restore_trainer


class BeetleTrainer:
    trains_discriminator = True

    def __init__(
        self,
        acoustic: AcousticModels,
        conditional: ConditionalModels,
        ema_latent_flow: nn.Module,
        config: TrainingConfig,
        adversarial: AdversarialConfig,
        runtime_seed: int,
        runtime: DistributedRuntime,
        optimizers: OptimizerSet,
        intervals: LoopIntervals,
        config_fingerprint: str,
        data_fingerprint: str,
        initial_loop: LoopState,
        input_builder: ConditionalInputBuilder,
    ) -> None:
        if conditional.audio_encoder is not acoustic.audio_encoder:
            raise ValueError("training requires one shared audio encoder")
        if conditional.f0_extractor is not acoustic.f0_extractor:
            raise ValueError("training requires one shared F0 extractor")
        self.acoustic = acoustic
        self.conditional = conditional
        self.ema_latent_flow = (
            ema_latent_flow.to(runtime.device).requires_grad_(False).eval()
        )
        self.config = config
        self.adversarial = adversarial
        self.runtime_seed = runtime_seed
        self.runtime = runtime
        self.world_size = runtime.world_size
        self.device = runtime.device
        self.optimizers = optimizers.prepare_distributed()
        self.intervals = intervals
        self.config_fingerprint = config_fingerprint
        self.data_fingerprint = data_fingerprint
        self._loop = initial_loop
        self.skipped_steps = 0
        self.input_builder = input_builder
        self.schedules = TrainingSchedules.from_config(config)
        self.accumulation_steps = config.accumulation_steps
        prepare_training_modules(self.acoustic, self.conditional, self.runtime)

    def loop_state(self) -> LoopState:
        return self._loop

    def set_loop_state(self, state: LoopState) -> None:
        self._loop = state

    def discriminator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        segment = self._segment(frame_mask, "discriminator")
        real = segment.samples(waveform)
        predicted_ratio = self.schedules.predicted_acoustic_ratio(
            self._loop.optimizer_step
        )
        with torch.no_grad(), self.runtime.autocast():
            target = self.acoustic.acoustic_targets(mel, frame_mask)
            posterior = self._synthesize_posterior(
                mel,
                frame_mask,
                segment,
                target,
                predicted_ratio,
                "discriminator",
            )
        with self.runtime.autocast():
            loss = discriminator_step_loss(
                self.acoustic.discriminators,
                real,
                posterior.waveform,
            )
            weights = self.schedules.acoustic_weights(self._loop.optimizer_step)
            weighted = loss * weights.discriminator
        self.optimizers.group("discriminator").backward(
            weighted / self.accumulation_steps
        )
        return (
            tensor_metric("discriminator", loss),
            tensor_metric("discriminator_total", weighted),
        )

    def generator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        target = self.input_builder.acoustic_targets(self.conditional, batch)
        acoustic = self._acoustic_backward(waveform, mel, frame_mask, target.features)
        inputs = self.input_builder.build(self.conditional, batch, self._loop, target)
        with self.runtime.autocast():
            conditional_losses = compute_conditional_losses(
                self.conditional, self.ema_latent_flow, inputs
            )
            conditional_total = conditional_losses.total(
                self.schedules.conditional_weights(self._loop.optimizer_step)
            )
        self.optimizers.group("generator").backward(
            conditional_total / self.accumulation_steps
        )
        names = self.schedules.conditional_names
        return (
            TrainingMetric("acoustic_prediction_ratio", acoustic.prediction_ratio),
            tensor_metric("encoder_kl", acoustic.encoder_kl),
            tensor_metric("f0", acoustic.f0),
            tensor_metric("n", acoustic.n),
            tensor_metric("posterior_reconstruction", acoustic.reconstruction),
            tensor_metric("generator_adversarial", acoustic.adversarial),
            tensor_metric("feature_matching", acoustic.feature_matching),
            *(
                tensor_metric(name, value)
                for name, value in zip(
                    names,
                    conditional_losses.values(),
                    strict=True,
                )
            ),
            tensor_metric(
                "generator_total",
                acoustic.total + conditional_total.detach(),
            ),
            *(
                tensor_metric(f"conditioning/{name}", value)
                for name, value in inputs.batch_statistics.named_values()
            ),
        )

    def _acoustic_backward(
        self,
        waveform: Tensor,
        mel: Tensor,
        frame_mask: Tensor,
        target: AcousticFeatures,
    ) -> AcousticBackwardMetrics:
        segment = self._segment(frame_mask, "generator")
        real = segment.samples(waveform)
        f0_target = segment.frames(target.f0)
        n_target = segment.frames(target.n)
        segment_frame_mask = segment.frames(frame_mask)
        predicted_ratio = self.schedules.predicted_acoustic_ratio(
            self._loop.optimizer_step
        )
        with self.runtime.autocast():
            posterior = self._synthesize_posterior(
                mel,
                frame_mask,
                segment,
                target,
                predicted_ratio,
                "generator",
            )
            encoder_kl = masked_kl_standard_normal(
                posterior.posterior.mean,
                posterior.posterior.log_scale,
                posterior.posterior.mask,
            )
            f0 = masked_f0_smooth_l1(
                posterior.acoustic.f0,
                f0_target,
                segment_frame_mask,
                self.acoustic.feature_linear.config.f0_scale_hz,
            )
            n = masked_n_smooth_l1(
                posterior.acoustic.n,
                n_target,
                segment_frame_mask,
            )
            posterior_reconstruction = self.acoustic.reconstruction_loss(
                posterior.waveform, real, posterior.sample_mask
            ).total
            weights = self.schedules.acoustic_weights(self._loop.optimizer_step)
            adversarial_view = generator_step_loss(
                self.acoustic.discriminators,
                real,
                posterior.waveform,
            )
            adversarial = adversarial_view.adversarial
            feature_matching = adversarial_view.feature_matching
            acoustic_total = (
                encoder_kl * weights.encoder_kl
                + f0 * weights.f0
                + n * weights.n
                + posterior_reconstruction * weights.reconstruction
                + adversarial * weights.generator_adversarial
                + feature_matching * weights.feature_matching
            )
        self.optimizers.group("generator").backward(
            acoustic_total / self.accumulation_steps
        )
        return AcousticBackwardMetrics(
            predicted_ratio,
            encoder_kl.detach(),
            f0.detach(),
            n.detach(),
            posterior_reconstruction.detach(),
            adversarial.detach(),
            feature_matching.detach(),
            acoustic_total.detach(),
        )

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]:
        metrics = self.optimizers.step(
            optimizer_step, diagnostics_due(optimizer_step + 1)
        )
        online_flow = self.runtime.unwrap(self.conditional.latent_flow)
        update_latent_flow_ema(
            self.ema_latent_flow, online_flow, online_flow.config.ema_decay
        )
        metrics = (*metrics, TrainingMetric("skipped_steps", float(self.skipped_steps)))
        self.skipped_steps = 0
        return metrics

    def discard_step(self) -> None:
        for group in self.optimizers.groups:
            group.optimizer.zero_grad(set_to_none=True)
        self.skipped_steps += 1

    def reduce_metrics(
        self, metrics: tuple[TrainingMetric, ...]
    ) -> tuple[TrainingMetric, ...]:
        return self.runtime.reduce_metrics(metrics)

    def checkpoint_payload(
        self,
        loop: LoopState,
        sampler_state: DataPipelineState,
        reporting: ReportingState,
    ) -> CheckpointPayload:
        return checkpoint_payload(self, loop, sampler_state, reporting)

    def restore(
        self,
        payload: CheckpointPayload,
        reset_optimizers: bool,
    ) -> DataPipelineState:
        return restore_trainer(self, payload, reset_optimizers)

    def _segment(self, frame_mask: Tensor, view: str) -> AlignedSegments:
        generator = self._generator(view, "segment")
        frame_count = self.adversarial.segment_samples // self.acoustic.output_hop
        lengths = frame_mask[:, 0].sum(dim=1).clamp_min(frame_count)
        positions = torch.arange(frame_mask.shape[-1], device=frame_mask.device)
        available = positions.view(1, 1, -1) < lengths.view(-1, 1, 1)
        return AlignedSegments.random(
            available,
            frame_count,
            self.acoustic.latent_downsample_rate,
            self.acoustic.output_hop,
            generator,
        )

    def _synthesize_posterior(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        segment: AlignedSegments,
        target: AcousticFeatures,
        predicted_ratio: float,
        view: str,
    ) -> AcousticSynthesis:
        return synthesize_training_posterior(
            self.acoustic,
            mel,
            frame_mask,
            segment,
            target,
            predicted_ratio,
            self._generator(view, "latent"),
            self._generator(view, "source"),
        )

    def _generator(self, view: str, purpose: str) -> torch.Generator:
        seed = derive_seed(
            self.runtime_seed,
            self._loop.cycle,
            self._loop.batch_index,
            view,
            purpose,
        )
        return torch.Generator(device=self.device).manual_seed(seed)

    def _inputs(self, batch: BeetleBatch) -> tuple[Tensor, Tensor, Tensor]:
        waveform = batch.waveform.to(self.device, non_blocking=True)
        mel = batch.mel.to(self.device, non_blocking=True)
        return waveform, mel, batch.frame_mask.to(self.device, non_blocking=True)
