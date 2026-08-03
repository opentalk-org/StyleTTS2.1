import logging

import torch
import torch.nn.functional as F

from ..config import TrainingConfig
from ..data import TrainingBatch
from ..gradient_sync import synchronize_gradients
from ..profiling import set_profiling_step
from ..setup import TrainingRuntime
from ...stages import StyleSource, TrainingLoss, stage_for_step
from .adversarial_training import discriminator_step, prosody_generator_loss
from .factorization_training import nuisance_losses
from .stage_requirements import requires_voice
from .training_forward import ForwardOutput, model_forward
from .validation_batch import styletts_zs_reconstruction_loss
from .rvq_health import check_rvq_health


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: TrainingConfig, runtime: TrainingRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self.alpha_flow_start = 0
        for training_stage in config.training_stages:
            if TrainingLoss.ALPHA_FLOW in training_stage.enabled_losses:
                break
            self.alpha_flow_start += training_stage.steps
        self.running_std: list[float] = []
        self.skipped_steps = 0
        self.step = 0

    def set_training_mode(self) -> None:
        stage = stage_for_step(self.config.training_stages, self.step)
        training_modules = {module.value for module in stage.trainable_modules}
        if stage.train_discriminators:
            training_modules.update(("msd", "mpd"))
        if TrainingLoss.PROSODY_ADVERSARIAL in stage.enabled_losses:
            training_modules.update(("prosody_discriminator", "duration_discriminator"))
        if TrainingLoss.SLM_ADVERSARIAL in stage.enabled_losses:
            training_modules.add("wd")
        self.runtime.models.set_training_mode(training_modules)

    def train_step(self, batch: TrainingBatch) -> dict[str, torch.Tensor | float]:
        set_profiling_step(self.step)
        batch = batch.to(self.runtime.accelerator.device)
        modules = self.runtime.models.modules
        accelerator = self.runtime.accelerator
        optimizer = self.runtime.optimizer
        stage = stage_for_step(self.config.training_stages, self.step)
        enabled = set(stage.enabled_losses)
        trainable = tuple(module.value for module in stage.trainable_modules)
        for name, module in modules.items():
            module.requires_grad_(name in trainable)
        modules.mpd.requires_grad_(stage.train_discriminators)
        modules.msd.requires_grad_(stage.train_discriminators)
        prosody_adversarial = TrainingLoss.PROSODY_ADVERSARIAL in enabled
        modules.prosody_discriminator.requires_grad_(prosody_adversarial)
        modules.duration_discriminator.requires_grad_(prosody_adversarial)
        modules.wd.requires_grad_(TrainingLoss.SLM_ADVERSARIAL in enabled)
        with accelerator.autocast():
            output = model_forward(
                self.runtime,
                batch,
                stage,
                max(0, self.step - self.alpha_flow_start),
                int(stage.max_decoder_seconds * 80),
            )
        (
            discriminator_loss,
            prosody_discriminator_loss,
            slm_discriminator_loss,
        ) = discriminator_step(
            self.runtime,
            output,
            batch,
            stage.train_discriminators,
            prosody_adversarial,
            TrainingLoss.SLM_ADVERSARIAL in enabled,
        )
        optimizer.zero_grad()
        losses = self._generator_losses(output, batch, enabled)
        style_batch_std = output.style_target.mean(-1).std(
            0,
            unbiased=False,
        ).mean()
        if stage.style_source is StyleSource.QUANTIZED:
            check_rvq_health(self.running_std, style_batch_std)
        else:
            self.running_std.clear()
        total = self._weighted_total(losses, stage.loss_weights, enabled)
        finite = bool(torch.isfinite(total).item())
        if finite:
            accelerator.backward(total)
        else:
            self.skipped_steps += 1
            logger.warning("non-finite generator loss at step=%s", self.step)
            for name in trainable:
                for parameter in modules[name].parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        for name in ("msd", "mpd"):
            modules[name].requires_grad_(stage.train_discriminators)
        for name in ("prosody_discriminator", "duration_discriminator"):
            modules[name].requires_grad_(prosody_adversarial)
        synchronize_gradients(accelerator, modules, trainable)
        for name in trainable:
            optimizer.step(name)
        return self._reported_metrics(
            output,
            losses,
            enabled,
            total,
            discriminator_loss,
            prosody_discriminator_loss,
            slm_discriminator_loss,
            style_batch_std,
            finite,
        )

    def _generator_losses(
        self,
        output: ForwardOutput,
        batch: TrainingBatch,
        enabled: set[TrainingLoss],
    ) -> dict[str, torch.Tensor]:
        zero = output.reconstructed.new_zeros(())
        losses = {item.value: zero for item in TrainingLoss}
        if TrainingLoss.MEL in enabled:
            losses["mel"] = self.runtime.losses.stft(
                output.reconstructed,
                output.waveform,
            )
        if TrainingLoss.F0 in enabled:
            losses["f0"] = styletts_zs_reconstruction_loss(
                output.target_f0,
                output.predicted_f0,
                batch.mel_lengths,
                divisor=10,
            )
        if TrainingLoss.NORM in enabled:
            losses["norm"] = styletts_zs_reconstruction_loss(
                output.target_energy,
                output.predicted_energy,
                batch.mel_lengths,
            )
        if enabled & {TrainingLoss.DURATION, TrainingLoss.DURATION_CE}:
            duration, duration_ce = self._duration_losses(output, batch)
            losses["duration"] = duration
            losses["duration_ce"] = duration_ce
        if enabled & {
            TrainingLoss.SEQUENCE_ALIGNMENT,
            TrainingLoss.MONOTONIC_ALIGNMENT,
        }:
            sequence, monotonic = self._alignment_losses(output, batch)
            losses["sequence_alignment"] = sequence
            losses["monotonic_alignment"] = monotonic
        if TrainingLoss.RVQ in enabled:
            losses["rvq"] = output.rvq_loss
        if TrainingLoss.ALPHA_FLOW in enabled:
            losses["alpha_flow"] = output.alpha_flow_loss
        if TrainingLoss.ADVERSARIAL in enabled:
            period, period_feature, period_generator, period_relative = (
                self.runtime.losses.generator.components(
                    output.waveform,
                    output.reconstructed,
                    self.runtime.models.modules.mpd,
                )
            )
            scale, scale_feature, scale_generator, scale_relative = (
                self.runtime.losses.generator.components(
                    output.waveform,
                    output.reconstructed,
                    self.runtime.models.modules.msd,
                )
            )
            losses["adversarial"] = period + scale
            losses["feature_matching"] = period_feature + scale_feature
            losses["generator_adversarial"] = period_generator + scale_generator
            losses["relative_adversarial"] = period_relative + scale_relative
        if TrainingLoss.WAVLM in enabled:
            losses["wavlm"] = self.runtime.losses.wavlm(
                output.waveform.detach().squeeze(1),
                output.reconstructed.squeeze(1),
            ).mean()
        if TrainingLoss.SLM_ADVERSARIAL in enabled:
            losses["slm_adversarial"] = self.runtime.losses.wavlm.generator(
                output.reconstructed.squeeze(1),
            )
        if TrainingLoss.PROSODY_ADVERSARIAL in enabled:
            (
                losses["prosody_adversarial"],
                losses["prosody_generator_adversarial"],
                losses["prosody_feature_matching"],
            ) = prosody_generator_loss(self.runtime, output, batch)
        if enabled & {TrainingLoss.VOICE_PAIR, TrainingLoss.VOICE_METRIC}:
            pair, metric = self.runtime.models.modules.factorization.identity_losses(
                output.voice,
                output.reference_voice,
                batch.speaker_ids,
            )
            losses["voice_pair"] = pair
            losses["voice_metric"] = metric
        if enabled & {TrainingLoss.VOICE_NUISANCE, TrainingLoss.STYLE_NUISANCE}:
            voice_nuisance, style_nuisance = nuisance_losses(
                self.runtime,
                output,
                batch,
                min(1.0, self.step / 1000),
            )
            losses["voice_nuisance"] = voice_nuisance
            losses["style_nuisance"] = style_nuisance
        if TrainingLoss.XCOV in enabled:
            losses["xcov"] = self.runtime.models.modules.factorization.cross_covariance(
                output.voice,
                output.style_target,
            )
        return losses

    def _reported_metrics(
        self,
        output: ForwardOutput,
        losses: dict[str, torch.Tensor],
        enabled: set[TrainingLoss],
        total: torch.Tensor,
        discriminator_loss: torch.Tensor,
        prosody_discriminator_loss: torch.Tensor,
        slm_discriminator_loss: torch.Tensor,
        style_batch_std: torch.Tensor,
        finite: bool,
    ) -> dict[str, torch.Tensor | float]:
        metrics: dict[str, torch.Tensor | float] = {
            item.value: losses[item.value] for item in enabled
        }
        metrics["total"] = total.detach()
        if TrainingLoss.ADVERSARIAL in enabled:
            metrics["discriminator"] = discriminator_loss
            metrics["feature_matching"] = losses["feature_matching"]
            metrics["generator_adversarial"] = losses[
                "generator_adversarial"
            ]
            metrics["relative_adversarial"] = losses[
                "relative_adversarial"
            ]
        if TrainingLoss.PROSODY_ADVERSARIAL in enabled:
            metrics["prosody_discriminator"] = prosody_discriminator_loss
            metrics["prosody_generator_adversarial"] = losses[
                "prosody_generator_adversarial"
            ]
            metrics["prosody_feature_matching"] = losses[
                "prosody_feature_matching"
            ]
        if TrainingLoss.SLM_ADVERSARIAL in enabled:
            metrics["slm_discriminator"] = slm_discriminator_loss
        if requires_voice(enabled):
            metrics["voice_batch_std"] = output.voice.std(
                0,
                unbiased=False,
            ).mean()
        style_diagnostics = {
            TrainingLoss.ALPHA_FLOW,
            TrainingLoss.RVQ,
            TrainingLoss.STYLE_NUISANCE,
            TrainingLoss.XCOV,
        }
        if enabled & style_diagnostics:
            metrics["style_batch_std"] = style_batch_std
        if enabled & {
            TrainingLoss.STYLE_NUISANCE,
            TrainingLoss.XCOV,
        }:
            with torch.no_grad():
                projected = self.runtime.models.modules.factorization.style_projection(
                    output.style_target.mean(-1)
                )
            metrics["style_projection_batch_std"] = projected.std(
                0,
                unbiased=False,
            ).mean()
        if TrainingLoss.ALPHA_FLOW in enabled:
            alpha_flow = self.runtime.models.modules.alpha_flow
            metrics.update(
                {
                    "alpha_flow_style_scale": alpha_flow.style_scale.detach().clone(),
                    "alpha_flow_style_scale_updates": (
                        alpha_flow.style_scale_updates.detach().clone()
                    ),
                    "alpha_flow_raw_mse": alpha_flow.last_raw_mse.detach().clone(),
                    "alpha_flow_velocity_cosine": (
                        alpha_flow.last_velocity_cosine.detach().clone()
                    ),
                }
            )
        metrics["step_skipped"] = float(not finite)
        metrics["skipped_steps"] = float(self.skipped_steps)
        return metrics

    @staticmethod
    def _duration_losses(output: ForwardOutput, batch: TrainingBatch) -> tuple[torch.Tensor, torch.Tensor]:
        duration = output.reconstructed.new_zeros(())
        cross_entropy = output.reconstructed.new_zeros(())
        items = zip(output.duration_predictions, output.duration_targets, batch.input_lengths, strict=True)
        for prediction, target, length in items:
            prediction = prediction[:length]
            target = target[:length].long()
            positions = torch.arange(prediction.size(1), device=prediction.device)
            binary = (positions[None] < target[:, None]).to(prediction.dtype)
            duration = duration + F.l1_loss(
                torch.sigmoid(prediction).sum(1)[1 : length - 1],
                target[1 : length - 1],
            )
            cross_entropy = cross_entropy + F.binary_cross_entropy_with_logits(prediction, binary)
        return duration / batch.texts.size(0), cross_entropy / batch.texts.size(0)

    @staticmethod
    def _alignment_losses(output: ForwardOutput, batch: TrainingBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = output.reconstructed.new_zeros(())
        for prediction, target, length in zip(
            output.alignment_predictions,
            batch.texts,
            batch.input_lengths,
            strict=True,
        ):
            sequence = sequence + F.cross_entropy(prediction[:length], target[:length])
        sequence = sequence / batch.texts.size(0)
        monotonic = F.l1_loss(output.soft_alignment, output.monotonic_alignment) * 10
        return sequence, monotonic

    @staticmethod
    def _weighted_total(losses, weights, enabled: set[TrainingLoss]) -> torch.Tensor:
        total = losses["mel"].new_zeros(())
        for item in enabled:
            total = total + losses[item.value] * getattr(weights, item.value)
        return total
