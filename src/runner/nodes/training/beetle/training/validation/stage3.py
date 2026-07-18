from uuid import UUID

import torch
from torch import Tensor, nn

from ...config.training import StageConfig
from ...data.sampling import derive_seed
from ...data.validation_types import ValidationRecording
from ...losses.acoustic import masked_f0_mse, masked_kl_standard_normal, masked_n_mse
from ...losses.adversarial import discriminator_step_loss, generator_step_loss
from ...models.model import Stage1Models, Stage1Synthesis
from ...models.stage2 import Stage2Models
from ..loss_schedules import Stage3Schedules
from ..reporting import TrainingMetric
from ..stage2_setup import Stage2InputBuilder
from ..state import StageKind
from .conditional import Stage2ValidationEvaluator
from .types import ValidationSampleResult


class Stage3ValidationEvaluator(Stage2ValidationEvaluator):
    stage = StageKind.STAGE3

    def __init__(
        self,
        stage1: Stage1Models,
        models: Stage2Models,
        ema_latent_flow: nn.Module,
        input_builder: Stage2InputBuilder,
        stage_config: StageConfig,
        runtime_seed: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            stage1,
            models,
            ema_latent_flow,
            input_builder,
            stage_config,
            runtime_seed,
            device,
        )
        self.stage3_schedules = Stage3Schedules.from_config(stage_config)

    @staticmethod
    def required_model_names() -> tuple[str, ...]:
        return (*Stage2ValidationEvaluator.required_model_names(), "discriminators")

    def modules(self) -> tuple[nn.Module, ...]:
        return (*super().modules(), self.stage1.discriminators)

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ValidationSampleResult, ...]:
        conditional = super().evaluate_samples(recordings, step)
        return tuple(
            self._combined_sample(recording, prediction, step)
            for recording, prediction in zip(recordings, conditional, strict=True)
        )

    def _combined_sample(
        self,
        recording: ValidationRecording,
        conditional: ValidationSampleResult,
        step: int,
    ) -> ValidationSampleResult:
        values = recording.batch.to(self.device)
        posterior = self._posterior(recording.audio_file_id, values, step)
        targets = self.stage1.acoustic_targets(values.mel, values.frame_mask)
        conditional_waveform = conditional.prediction.unsqueeze(0).to(self.device)
        conditional_f0 = conditional.f0[1].unsqueeze(0).to(self.device)
        conditional_n = conditional.n[1].unsqueeze(0).to(self.device)
        encoder_kl = masked_kl_standard_normal(
            posterior.posterior.mean,
            posterior.posterior.log_scale,
            posterior.posterior.mask,
        )
        f0 = 0.5 * (
            masked_f0_mse(posterior.acoustic.f0, targets.f0, posterior.decoded.mask)
            + masked_f0_mse(conditional_f0, targets.f0, posterior.decoded.mask)
        )
        n = 0.5 * (
            masked_n_mse(posterior.acoustic.n, targets.n, posterior.decoded.mask)
            + masked_n_mse(conditional_n, targets.n, posterior.decoded.mask)
        )
        reconstruction = 0.5 * (
            self.stage1.reconstruction_loss(
                posterior.waveform,
                values.waveform,
                posterior.sample_mask,
            ).total
            + self.stage1.reconstruction_loss(
                conditional_waveform,
                values.waveform,
                posterior.sample_mask,
            ).total
        )
        discriminator = 0.5 * (
            discriminator_step_loss(
                self.stage1.discriminators,
                values.waveform,
                posterior.waveform,
            )
            + discriminator_step_loss(
                self.stage1.discriminators,
                values.waveform,
                conditional_waveform,
            )
        )
        posterior_adversarial = generator_step_loss(
            self.stage1.discriminators,
            values.waveform,
            posterior.waveform,
        )
        conditional_adversarial = generator_step_loss(
            self.stage1.discriminators,
            values.waveform,
            conditional_waveform,
        )
        adversarial = 0.5 * (
            posterior_adversarial.adversarial
            + conditional_adversarial.adversarial
        )
        feature_matching = 0.5 * (
            posterior_adversarial.feature_matching
            + conditional_adversarial.feature_matching
        )
        weights = self.stage3_schedules.weights(step)
        stage2_metrics = tuple(
            metric for metric in conditional.losses if metric.name != "stage2_total"
        )
        flow_total = next(
            metric.value
            for metric in conditional.losses
            if metric.name == "stage2_total"
        )
        acoustic_total = (
            encoder_kl * weights.encoder_kl
            + f0 * weights.f0
            + n * weights.n
            + reconstruction * weights.reconstruction
            + adversarial * weights.generator_adversarial
            + feature_matching * weights.feature_matching
        )
        generator_total = acoustic_total + flow_total
        losses = (
            _metric("encoder_kl", encoder_kl),
            _metric("f0", f0),
            _metric("n", n),
            _metric("reconstruction", reconstruction),
            _metric("discriminator", discriminator),
            _metric("generator_adversarial", adversarial),
            _metric("feature_matching", feature_matching),
            *stage2_metrics,
            _metric("discriminator_total", discriminator * weights.discriminator),
            _metric("generator_total", generator_total),
        )
        return ValidationSampleResult(
            recording.audio_file_id,
            losses,
            conditional.ground_truth,
            conditional.prediction,
            conditional.latent,
            conditional.f0,
            conditional.n,
            conditional.mel,
            conditional.alignment,
        )

    def _posterior(
        self,
        audio_file_id: UUID,
        batch: object,
        step: int,
    ) -> Stage1Synthesis:
        latent = self._generator(step, audio_file_id, "posterior-latent")
        source = self._generator(step, audio_file_id, "posterior-source")
        return self.stage1.reconstruct(batch.mel, batch.frame_mask, latent, source)

    def _generator(
        self,
        step: int,
        audio_file_id: UUID,
        view: str,
    ) -> torch.Generator:
        seed = derive_seed(
            self.runtime_seed,
            self.stage,
            step,
            audio_file_id,
            view,
        )
        return torch.Generator(device=self.device).manual_seed(seed)


def _metric(name: str, value: Tensor | float) -> TrainingMetric:
    scalar = float(value.detach().cpu()) if isinstance(value, Tensor) else value
    return TrainingMetric(name, scalar)
