from uuid import UUID

import torch
from torch import Tensor, nn

from ...config.training import TrainingConfig
from ...data.sampling import derive_seed
from ...data.validation_records import ValidationRecording
from ...losses.acoustic import (
    masked_f0_smooth_l1,
    masked_kl_standard_normal,
    masked_n_smooth_l1,
)
from ...losses.adversarial import discriminator_step_loss, generator_step_loss
from ...models.model import AcousticModels, AcousticSynthesis
from ...models.conditional import ConditionalModels
from ..loss_schedules import TrainingSchedules
from ..reporting import TrainingMetric
from ..setup import ConditionalInputBuilder
from .conditional import ConditionalValidationEvaluator
from .types import (
    ConditionalValidationSample,
    ValidationArtifactSet,
    ValidationSampleResult,
    trim_signal_pair,
    trim_waveform_pair,
)


class TrainingValidationEvaluator(ConditionalValidationEvaluator):
    def __init__(
        self,
        acoustic: AcousticModels,
        models: ConditionalModels,
        ema_latent_flow: nn.Module,
        input_builder: ConditionalInputBuilder,
        training_config: TrainingConfig,
        runtime_seed: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            acoustic,
            models,
            ema_latent_flow,
            input_builder,
            training_config,
            runtime_seed,
            device,
        )
        self.training_schedules = TrainingSchedules.from_config(training_config)

    @staticmethod
    def required_model_names() -> tuple[str, ...]:
        return (*ConditionalValidationEvaluator.required_model_names(), "discriminators")

    def modules(self) -> tuple[nn.Module, ...]:
        return (*super().modules(), self.acoustic.discriminators)

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
        conditional: ConditionalValidationSample,
        step: int,
    ) -> ValidationSampleResult:
        values = recording.batch.to(self.device)
        posterior = self._posterior(recording.audio_file_id, values, step)
        full = conditional.artifacts
        real_waveform = full.ground_truth.unsqueeze(0).to(self.device)
        sample_count = real_waveform.shape[-1]
        posterior_waveform = posterior.waveform[:, :, :sample_count]
        sample_mask = posterior.sample_mask[:, :, :sample_count]
        target_f0 = full.f0[0].unsqueeze(0).to(self.device)
        frame_count = target_f0.shape[-1]
        posterior_f0 = posterior.acoustic.f0[:, :frame_count]
        frame_mask = posterior.decoded.mask[:, :, :frame_count]
        target_n = full.n[0].unsqueeze(0).to(self.device)
        posterior_n = posterior.acoustic.n[:, :frame_count]
        encoder_kl = masked_kl_standard_normal(
            posterior.posterior.mean,
            posterior.posterior.log_scale,
            posterior.posterior.mask,
        )
        f0 = masked_f0_smooth_l1(
            posterior_f0,
            target_f0,
            frame_mask,
            self.acoustic.feature_linear.config.f0_scale_hz,
        )
        n = masked_n_smooth_l1(
            posterior_n,
            target_n,
            frame_mask,
        )
        reconstruction = self.acoustic.reconstruction_loss(
            posterior_waveform,
            real_waveform,
            sample_mask,
        ).total
        discriminator = discriminator_step_loss(
            self.acoustic.discriminators,
            real_waveform,
            posterior_waveform,
        )
        adversarial_view = generator_step_loss(
            self.acoustic.discriminators,
            real_waveform,
            posterior_waveform,
        )
        adversarial = adversarial_view.adversarial
        feature_matching = adversarial_view.feature_matching
        weights = self.training_schedules.acoustic_weights(step)
        conditional_metrics = tuple(
            metric for metric in conditional.losses if metric.name != "conditional_total"
        )
        flow_total = next(
            metric.value
            for metric in conditional.losses
            if metric.name == "conditional_total"
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
            *conditional_metrics,
            _metric("discriminator_total", discriminator * weights.discriminator),
            _metric("generator_total", generator_total),
        )
        _, posterior_prediction = trim_waveform_pair(
            real_waveform,
            posterior_waveform,
            sample_count,
        )
        target_mel, posterior_mel = self._artifact_mels(
            real_waveform,
            posterior_waveform,
        )
        audio = ValidationArtifactSet(
            full.ground_truth,
            posterior_prediction,
            _cpu(posterior.posterior.latent[0]),
            trim_signal_pair(target_f0[0], posterior_f0[0], frame_count),
            trim_signal_pair(target_n[0], posterior_n[0], frame_count),
            (_cpu(target_mel[0]), _cpu(posterior_mel[0])),
            full.alignment,
        )
        return ValidationSampleResult(
            recording.audio_file_id,
            losses,
            full,
            audio,
            conditional.seed,
        )

    def _posterior(
        self,
        audio_file_id: UUID,
        batch: object,
        step: int,
    ) -> AcousticSynthesis:
        latent = self._generator(step, audio_file_id, "posterior-latent")
        source = self._generator(step, audio_file_id, "posterior-source")
        return self.acoustic.reconstruct(batch.mel, batch.frame_mask, latent, source)

    def _generator(
        self,
        step: int,
        audio_file_id: UUID,
        view: str,
    ) -> torch.Generator:
        seed = derive_seed(
            self.runtime_seed,
            step,
            audio_file_id,
            view,
        )
        return torch.Generator(device=self.device).manual_seed(seed)


def _metric(name: str, value: Tensor | float) -> TrainingMetric:
    scalar = float(value.detach().cpu()) if isinstance(value, Tensor) else value
    return TrainingMetric(name, scalar)


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().clone()
