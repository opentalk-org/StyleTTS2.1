import torch
from torch import Tensor, nn

from ..config.training import AdversarialConfig, TrainingConfig
from ..data.pipeline import DataPipelineState
from ..data.records import BeetleBatch
from ..losses.adversarial import discriminator_step_loss
from ..losses.conditional import compute_conditional_losses
from ..models.model import AcousticModels
from ..models.conditional import ConditionalModels
from .callbacks import TrainingMetric
from .checkpoint import CheckpointPayload
from .acoustic_synthesis import (
    AcousticBackwardMetrics,
    AcousticTrainingView,
    acoustic_backward,
    batch_inputs,
    build_acoustic_training_view,
)
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
        self._acoustic_view: AcousticTrainingView | None = None
        self._discriminator_optimizer_metrics: tuple[TrainingMetric, ...] | None = None
        prepare_training_modules(self.acoustic, self.conditional, self.runtime)

    def loop_state(self) -> LoopState:
        return self._loop

    def set_loop_state(self, state: LoopState) -> None:
        self._loop = state

    def discriminator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        segment = self._segment(frame_mask, "discriminator")
        real = segment.samples(waveform)
        with torch.no_grad(), self.runtime.autocast():
            posterior = self._synthesize_posterior(
                mel,
                frame_mask,
                segment,
                "discriminator",
            )
        self._acoustic_view = view
        real = view.segment.samples(waveform)
        with self.runtime.autocast():
            loss = discriminator_step_loss(
                self.acoustic.discriminators,
                real,
                view.synthesis.waveform,
            )
            weights = self.schedules.acoustic_weights(self._loop.optimizer_step)
            weighted = loss * weights.discriminator
        self.optimizers.group("discriminator").backward(
            weighted / self.accumulation_steps
        )
        if self.accumulation_steps == 1:
            self._discriminator_optimizer_metrics = self.optimizers.step_group(
                "discriminator",
                self._loop.optimizer_step,
                diagnostics_due(self._loop.optimizer_step + 1),
            )
        return (
            tensor_metric("discriminator", loss),
            tensor_metric("discriminator_total", weighted),
        )

    def generator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, _, _ = self._inputs(batch)
        target = self.input_builder.acoustic_targets(self.conditional, batch)
        if self._acoustic_view is None:
            raise RuntimeError("generator backward requires discriminator synthesis")
        acoustic = self._acoustic_backward(waveform, self._acoustic_view)
        self._acoustic_view = None
        inputs = self.input_builder.build(self.conditional, batch, self._loop, target)
        with self.runtime.autocast():
            conditional_losses = compute_conditional_losses(self.conditional, inputs)
            conditional_total = conditional_losses.total(
                self.schedules.conditional_weights(self._loop.optimizer_step)
            )
        self.optimizers.group("generator").backward(
            conditional_total / self.accumulation_steps
        )
        names = self.schedules.conditional_names
        return (
            TrainingMetric("f0_prediction_ratio", acoustic.f0_prediction_ratio),
            tensor_metric("encoder_kl", acoustic.encoder_kl),
            tensor_metric("f0", acoustic.f0),
            tensor_metric("n", acoustic.n),
            tensor_metric("posterior_reconstruction", acoustic.reconstruction),
            tensor_metric("generator_adversarial", acoustic.adversarial),
            tensor_metric("feature_period", acoustic.feature_period),
            tensor_metric("feature_resolution", acoustic.feature_resolution),
            tensor_metric("feature_matching", acoustic.feature_matching),
            tensor_metric("vocoder_total", acoustic.vocoder_total),
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
        view: AcousticTrainingView,
    ) -> AcousticBackwardMetrics:
        return acoustic_backward(
            self.acoustic,
            self.runtime,
            self.optimizers.group("generator"),
            self.accumulation_steps,
            waveform,
            view,
            self.schedules.acoustic_weights(self._loop.optimizer_step),
            self._generator("generator", "latent"),
            self._loop.optimizer_step + 1,
        )

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]:
        diagnostics = diagnostics_due(optimizer_step + 1)
        if self.accumulation_steps == 1:
            if self._discriminator_optimizer_metrics is None:
                raise RuntimeError("discriminator optimizer step is missing")
            metrics = (
                *self._discriminator_optimizer_metrics,
                *self.optimizers.step_group("generator", optimizer_step, diagnostics),
            )
            self._discriminator_optimizer_metrics = None
        else:
            metrics = self.optimizers.step(optimizer_step, diagnostics)
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
        self._acoustic_view = None
        self._discriminator_optimizer_metrics = None
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
        return training_segment(
            frame_mask,
            self.adversarial.segment_samples,
            self.acoustic,
            self._generator(view, "segment"),
        )

    def _synthesize_posterior(
        self,
        mel: Tensor,
        frame_mask: Tensor,
        segment: AlignedSegments,
        view: str,
    ) -> AcousticSynthesis:
        return synthesize_training_posterior(
            self.acoustic,
            mel,
            frame_mask,
            segment,
            self._generator(view, "latent"),
        )

    def _generator(self, view: str, purpose: str) -> torch.Generator:
        return training_generator(
            self.runtime_seed,
            self._loop,
            self.device,
            view,
            purpose,
        )

    def _inputs(self, batch: BeetleBatch) -> tuple[Tensor, Tensor, Tensor]:
        return batch_inputs(batch, self.device)
