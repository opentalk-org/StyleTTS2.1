import logging

import torch
import torch.nn.functional as F

from ..config import TrainingConfig
from ..data import TrainingBatch
from ..gradient_sync import synchronize_gradients
from ..profiling import set_profiling_step
from ..setup import TrainingRuntime
from ...stages import TrainingLoss, stage_for_step
from .factorization_training import style_nuisance_loss
from .training_forward import ForwardOutput, model_forward
from .validation_batch import styletts_zs_reconstruction_loss


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: TrainingConfig, runtime: TrainingRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self.alpha_flow_start = 0
        for training_stage in config.training_stages:
            if training_stage.loss_weights.alpha_flow > 0:
                break
            self.alpha_flow_start += training_stage.steps
        self.skipped_steps = 0
        self.step = 0

    def set_training_mode(self) -> None:
        stage = stage_for_step(self.config.training_stages, self.step)
        weights = stage.loss_weights
        training_modules = {module.value for module in stage.trainable_modules}
        if weights.adversarial > 0:
            training_modules.update(("msd", "mpd"))
        if weights.prosody_adversarial > 0:
            training_modules.update(("prosody_discriminator", "duration_discriminator"))
        if weights.slm_adversarial > 0:
            training_modules.add("wd")
        self.runtime.models.set_training_mode(training_modules)

    def train_step(self, batch: TrainingBatch) -> dict[str, torch.Tensor | float]:
        set_profiling_step(self.step)
        batch = batch.to(self.runtime.accelerator.device)
        modules = self.runtime.models.modules
        accelerator = self.runtime.accelerator
        optimizer = self.runtime.optimizer
        stage = stage_for_step(self.config.training_stages, self.step)
        weights = stage.loss_weights
        trainable = tuple(module.value for module in stage.trainable_modules)
        for name, module in modules.items():
            module.requires_grad_(name in trainable)
        waveform_adversarial = weights.adversarial > 0
        modules.mpd.requires_grad_(waveform_adversarial)
        modules.msd.requires_grad_(waveform_adversarial)
        prosody_adversarial = weights.prosody_adversarial > 0
        modules.prosody_discriminator.requires_grad_(prosody_adversarial)
        modules.duration_discriminator.requires_grad_(prosody_adversarial)
        modules.wd.requires_grad_(weights.slm_adversarial > 0)
        with accelerator.autocast():
            output = model_forward(
                self.runtime,
                batch,
                stage,
                max(0, self.step - self.alpha_flow_start),
                int(stage.max_decoder_seconds * 80),
            )
        discriminator_loss = output.reconstructed.new_zeros(())
        prosody_discriminator_loss = output.reconstructed.new_zeros(())
        slm_discriminator_loss = output.reconstructed.new_zeros(())

        if waveform_adversarial:
            for name, discriminator in (
                ("mpd", modules.mpd),
                ("msd", modules.msd),
            ):
                optimizer.zero_grad(name)
                with accelerator.autocast():
                    loss = self.runtime.losses.discriminator(
                        output.waveform.detach(),
                        output.reconstructed.detach(),
                        discriminator,
                    )
                accelerator.backward(loss)
                synchronize_gradients(accelerator, modules, (name,))
                optimizer.step(name)
                discriminator.requires_grad_(False)
                discriminator_loss = discriminator_loss + loss.detach()

        if prosody_adversarial:
            prosody_items = (
                (
                    "prosody_discriminator",
                    self.runtime.losses.prosody_discriminator,
                    output.prosody_fake,
                    output.prosody_real,
                    batch.mel_lengths.to(output.reconstructed.device) // 2,
                ),
                (
                    "duration_discriminator",
                    self.runtime.losses.duration_discriminator,
                    output.duration_fake,
                    output.duration_real,
                    batch.input_lengths.to(output.reconstructed.device),
                ),
            )
            for name, objective, fake, real, lengths in prosody_items:
                optimizer.zero_grad(name)
                loss = objective(
                    fake.float(),
                    real.float(),
                    output.style_target.float(),
                    lengths,
                    real.size(-1),
                )
                accelerator.backward(loss)
                synchronize_gradients(accelerator, modules, (name,))
                optimizer.step(name)
                modules[name].requires_grad_(False)
                prosody_discriminator_loss += loss.detach()

        if weights.slm_adversarial > 0:
            optimizer.zero_grad("wd")
            with accelerator.autocast():
                slm_discriminator_loss = self.runtime.losses.wavlm.discriminator(
                    output.waveform.detach().squeeze(1),
                    output.reconstructed.detach().squeeze(1),
                )
            accelerator.backward(slm_discriminator_loss)
            synchronize_gradients(accelerator, modules, ("wd",))
            optimizer.step("wd")
            modules.wd.requires_grad_(False)
            slm_discriminator_loss = slm_discriminator_loss.detach()

        for name in trainable:
            optimizer.zero_grad(name)
        with accelerator.autocast():
            zero = output.reconstructed.new_zeros(())
            losses = {item.value: zero for item in TrainingLoss}
            if weights.mel > 0:
                losses["mel"] = self.runtime.losses.stft(
                    output.reconstructed,
                    output.waveform,
                )
            if weights.f0 > 0:
                losses["f0"] = styletts_zs_reconstruction_loss(
                    output.target_f0,
                    output.predicted_f0,
                    batch.mel_lengths,
                    divisor=10,
                )
            if weights.norm > 0:
                losses["norm"] = styletts_zs_reconstruction_loss(
                    output.target_energy,
                    output.predicted_energy,
                    batch.mel_lengths,
                )
            if weights.duration > 0 or weights.duration_ce > 0:
                duration, duration_ce = self._duration_losses(output, batch)
                losses["duration"] = duration
                losses["duration_ce"] = duration_ce
            if (
                weights.sequence_alignment > 0
                or weights.monotonic_alignment > 0
            ):
                sequence, monotonic = self._alignment_losses(output, batch)
                losses["sequence_alignment"] = sequence
                losses["monotonic_alignment"] = monotonic
            if weights.rvq > 0:
                losses["rvq"] = output.rvq_loss
            if weights.alpha_flow > 0:
                losses["alpha_flow"] = output.alpha_flow_loss
            if weights.adversarial > 0:
                period, period_feature, period_generator, period_relative = (
                    self.runtime.losses.generator.components(
                        output.waveform,
                        output.reconstructed,
                        modules.mpd,
                    )
                )
                scale, scale_feature, scale_generator, scale_relative = (
                    self.runtime.losses.generator.components(
                        output.waveform,
                        output.reconstructed,
                        modules.msd,
                    )
                )
                losses["adversarial"] = period + scale
                losses["feature_matching"] = period_feature + scale_feature
                losses["generator_adversarial"] = period_generator + scale_generator
                losses["relative_adversarial"] = period_relative + scale_relative
            if weights.wavlm > 0 or weights.slm_adversarial > 0:
                slm_adversarial, wavlm = self.runtime.losses.wavlm.generator(
                    output.waveform.detach().squeeze(1),
                    output.reconstructed.squeeze(1),
                    feature_matching=weights.wavlm > 0,
                    adversarial=weights.slm_adversarial > 0,
                )
                losses["wavlm"] = wavlm
                losses["slm_adversarial"] = slm_adversarial
            if weights.speaker_feature > 0 or weights.speaker_similarity > 0:
                speaker_feature, speaker_similarity = (
                    self.runtime.losses.speaker_verification(
                        output.waveform.detach(),
                        output.reconstructed,
                    )
                )
                losses["speaker_feature"] = speaker_feature
                losses["speaker_similarity"] = speaker_similarity
            if weights.prosody_adversarial > 0:
                prosody, prosody_features = self.runtime.losses.prosody_generator(
                    output.prosody_fake.float(),
                    output.prosody_real.float(),
                    output.style_target.float(),
                    batch.mel_lengths.to(output.reconstructed.device) // 2,
                    output.prosody_real.size(-1),
                )
                duration, duration_features = self.runtime.losses.duration_generator(
                    output.duration_fake.float(),
                    output.duration_real.float(),
                    output.style_target.float(),
                    batch.input_lengths.to(output.reconstructed.device),
                    output.duration_real.size(-1),
                )
                adversarial = prosody + duration
                feature_matching = prosody_features + duration_features
                losses["prosody_generator_adversarial"] = adversarial
                losses["prosody_feature_matching"] = feature_matching
                losses["prosody_adversarial"] = adversarial + feature_matching
            if weights.style_nuisance > 0:
                losses["style_nuisance"] = style_nuisance_loss(
                    self.runtime,
                    output,
                    batch,
                    min(1.0, self.step / 1000),
                )
            if weights.xcov > 0:
                losses["xcov"] = modules.factorization.cross_covariance(
                    output.voice,
                    output.style_target,
                )
        style_batch_std = output.style_target.mean(-1).std(
            0,
            unbiased=False,
        ).mean()
        total = self._weighted_total(losses, weights)
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
        synchronize_gradients(accelerator, modules, trainable)
        for name in trainable:
            optimizer.step(name)
        return self._reported_metrics(
            output,
            losses,
            weights,
            total,
            discriminator_loss,
            prosody_discriminator_loss,
            slm_discriminator_loss,
            style_batch_std,
            finite,
        )

    def _reported_metrics(
        self,
        output: ForwardOutput,
        losses: dict[str, torch.Tensor],
        weights,
        total: torch.Tensor,
        discriminator_loss: torch.Tensor,
        prosody_discriminator_loss: torch.Tensor,
        slm_discriminator_loss: torch.Tensor,
        style_batch_std: torch.Tensor,
        finite: bool,
    ) -> dict[str, torch.Tensor | float]:
        metrics: dict[str, torch.Tensor | float] = {
            item.value: losses[item.value]
            for item in TrainingLoss
            if getattr(weights, item.value) > 0
        }
        metrics["total"] = total.detach()
        if weights.adversarial > 0:
            metrics["discriminator"] = discriminator_loss
            metrics["feature_matching"] = losses["feature_matching"]
            metrics["generator_adversarial"] = losses[
                "generator_adversarial"
            ]
            metrics["relative_adversarial"] = losses[
                "relative_adversarial"
            ]
        if weights.prosody_adversarial > 0:
            metrics["prosody_discriminator"] = prosody_discriminator_loss
            metrics["prosody_generator_adversarial"] = losses[
                "prosody_generator_adversarial"
            ]
            metrics["prosody_feature_matching"] = losses[
                "prosody_feature_matching"
            ]
        if weights.slm_adversarial > 0:
            metrics["slm_discriminator"] = slm_discriminator_loss
        if (
            weights.adversarial > 0
            or weights.mel > 0
            or weights.slm_adversarial > 0
            or weights.speaker_feature > 0
            or weights.speaker_similarity > 0
            or weights.style_nuisance > 0
            or weights.wavlm > 0
            or weights.xcov > 0
        ):
            metrics["voice_batch_std"] = output.voice.std(
                0,
                unbiased=False,
            ).mean()
        if (
            weights.alpha_flow > 0
            or weights.rvq > 0
            or weights.style_nuisance > 0
            or weights.xcov > 0
        ):
            metrics["style_batch_std"] = style_batch_std
        if weights.style_nuisance > 0 or weights.xcov > 0:
            with torch.no_grad():
                projected = self.runtime.models.modules.factorization.style_projection(
                    output.style_target.mean(-1)
                )
            metrics["style_projection_batch_std"] = projected.std(
                0,
                unbiased=False,
            ).mean()
        if weights.alpha_flow > 0:
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
    def _weighted_total(losses, weights) -> torch.Tensor:
        total = losses["mel"].new_zeros(())
        for item in TrainingLoss:
            total = total + losses[item.value] * getattr(weights, item.value)
        return total
