import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..config import TrainingConfig
from ..data import TrainingBatch
from ..gradient_sync import synchronize_gradients
from ..profiling import profiling_fn, set_profiling_step
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.running_std: list[float] = []
        self.skipped_steps = 0
        self.step = 0

    def set_training_mode(self) -> None:
        self.runtime.models.set_training_mode()

    def train_step(
        self,
        batch: TrainingBatch,
    ) -> dict[str, torch.Tensor | float]:
        set_profiling_step(self.step)
        with profiling_fn("data_to_device"):
            batch = batch.to(self.runtime.accelerator.device)
        modules = self.runtime.models.modules
        accelerator = self.runtime.accelerator
        predictor_core = accelerator.unwrap_model(modules.predictor)
        diffusion_core = accelerator.unwrap_model(modules.diffusion)
        model_config = self.runtime.models.parameters
        losses = self.runtime.losses
        optimizer = self.runtime.optimizer
        loss_config = self.config.loss_params

        with torch.no_grad():
            with profiling_fn("forward.mask_preparation"):
                mask = length_to_mask(
                    batch.mel_lengths // (2**self.runtime.models.n_down)
                ).to(batch.mels.device)
                text_mask = length_to_mask(batch.input_lengths).to(
                    batch.texts.device
                )
            style_reference = None
            if (
                model_config.multispeaker
                and self.step >= loss_config.diffusion_start_step
            ):
                with profiling_fn("forward.reference_style"):
                    reference_style = modules.style_encoder(
                        batch.reference_mels.unsqueeze(1)
                    )
                    reference_prosody = modules.predictor_encoder(
                        batch.reference_mels.unsqueeze(1)
                    )
                    style_reference = torch.cat(
                        [reference_style, reference_prosody],
                        dim=1,
                    )

        with accelerator.autocast():
            with profiling_fn("forward.text_alignment"):
                _, alignment_predictions, soft_alignment = modules.text_aligner(
                    batch.mels,
                    mask,
                    batch.texts,
                )
            with profiling_fn("forward.monotonic_alignment"):
                soft_alignment = soft_alignment.transpose(-1, -2)
                soft_alignment = soft_alignment[..., 1:].transpose(-1, -2)
                alignment_mask = mask_from_lens(
                    soft_alignment,
                    batch.input_lengths,
                    batch.mel_lengths // (2**self.runtime.models.n_down),
                )
                monotonic_alignment = maximum_path(
                    soft_alignment,
                    alignment_mask,
                )
            with profiling_fn("forward.text_encoder"):
                text_encoding = modules.text_encoder(
                    batch.texts,
                    batch.input_lengths,
                    text_mask,
                )
            selected_alignment = (
                soft_alignment
                if bool(random.getrandbits(1))
                else monotonic_alignment
            )
            aligned_text = text_encoding @ selected_alignment
            duration_targets = monotonic_alignment.sum(axis=-1).detach()

            duration_styles = []
            acoustic_styles = []
            with profiling_fn("forward.style_encoders"):
                for index, length in enumerate(batch.mel_lengths):
                    mel = batch.mels[index, :, :length]
                    duration_styles.append(
                        modules.predictor_encoder(
                            mel.unsqueeze(0).unsqueeze(1)
                        )
                    )
                    acoustic_styles.append(
                        modules.style_encoder(
                            mel.unsqueeze(0).unsqueeze(1)
                        )
                    )
            duration_style = torch.stack(duration_styles).squeeze(1)
            acoustic_style = torch.stack(acoustic_styles).squeeze(1)
            style_target = torch.cat(
                [acoustic_style, duration_style],
                dim=-1,
            ).detach()
            with profiling_fn("forward.bert"):
                bert = modules.bert(
                    batch.texts,
                    attention_mask=(~text_mask).int(),
                )
                duration_encoding = modules.bert_encoder(bert).transpose(-1, -2)

            style_loss: torch.Tensor | float = 0.0
            diffusion_loss: torch.Tensor | float = 0.0
            if self.step >= loss_config.diffusion_start_step:
                diffusion = diffusion_core.diffusion
                if model_config.diffusion.dist.estimate_sigma_data:
                    diffusion.sigma_data = (
                        style_target.std(axis=-1).mean().item()
                    )
                    self.running_std.append(diffusion.sigma_data)
                sampler_arguments = {
                    "noise": torch.randn_like(style_target).unsqueeze(1),
                    "embedding": bert,
                    "embedding_scale": 1,
                    "embedding_mask_proba": 0.1,
                    "num_steps": int(np.random.randint(3, 5)),
                }
                if style_reference is not None:
                    sampler_arguments["features"] = style_reference
                with profiling_fn("forward.diffusion"):
                    style_prediction = self.runtime.diffusion_sampler(
                        **sampler_arguments
                    ).squeeze(1)
                    if style_reference is None:
                        diffusion_loss = diffusion(
                            style_target.unsqueeze(1),
                            embedding=bert,
                        ).mean()
                    else:
                        diffusion_loss = modules.diffusion(
                            style_target.unsqueeze(1),
                            embedding=bert,
                            features=style_reference,
                        ).mean()
                style_loss = F.l1_loss(
                    style_prediction,
                    style_target.detach(),
                )

            crop_frames = min(
                int(batch.mel_lengths.min().item() / 2 - 1),
                self.config.max_len // 2,
            )
            aligned_crops = []
            crop_starts = []
            mel_crops = []
            waveform_crops = []
            with profiling_fn("data_crop_collection"):
                for index, mel_frames in enumerate(batch.mel_lengths):
                    half_frames = int(mel_frames.item() / 2)
                    start = int(np.random.randint(0, half_frames - crop_frames))
                    aligned_crops.append(
                        aligned_text[index, :, start : start + crop_frames]
                    )
                    crop_starts.append(start)
                    mel_crops.append(
                        batch.mels[
                            index,
                            :,
                            start * 2 : (start + crop_frames) * 2,
                        ]
                    )
                    samples = batch.waves[index][
                        start * 600 : (start + crop_frames) * 600
                    ]
                    waveform_crops.append(
                        torch.from_numpy(samples).to(batch.mels.device)
                    )
                aligned_crops = torch.stack(aligned_crops)
                mel_crops = torch.stack(mel_crops).detach()
                waveform = torch.stack(waveform_crops).float().detach().unsqueeze(1)
            with profiling_fn("forward.crop_style_encoders"):
                style = modules.style_encoder(mel_crops.unsqueeze(1))
                crop_duration_style = modules.predictor_encoder(
                    mel_crops.unsqueeze(1)
            )
            with torch.no_grad():
                with profiling_fn("forward.pitch_and_norm_targets"):
                    target_f0, _, _ = modules.pitch_extractor(
                        mel_crops.unsqueeze(1)
                    )
                    target_f0 = target_f0.squeeze(-1)
                    target_norm = log_norm(mel_crops.unsqueeze(1)).squeeze(1)
                with profiling_fn("forward.decoder_reference"):
                    reference_prediction = modules.decoder(
                        aligned_crops,
                        target_f0,
                        target_norm,
                        style,
                    )
            with profiling_fn("forward.predictor"):
                (
                    duration_predictions,
                    prosody,
                    predicted_f0,
                    predicted_norm,
                ) = modules.predictor(
                    duration_encoding,
                    duration_style,
                    batch.input_lengths,
                    monotonic_alignment,
                    text_mask,
                    crop_starts,
                    crop_frames,
                    crop_duration_style,
                )
            with profiling_fn("forward.decoder"):
                reconstructed = modules.decoder(
                    aligned_crops,
                    predicted_f0,
                    predicted_norm,
                    style,
                )
            f0_loss = F.smooth_l1_loss(target_f0, predicted_f0) / 10
            norm_loss = F.smooth_l1_loss(target_norm, predicted_norm)

        optimizer.zero_grad()
        with profiling_fn("discriminator"):
            with accelerator.autocast():
                with profiling_fn("forward"):
                    discriminator_loss = losses.discriminator(
                        waveform.detach(),
                        reconstructed.detach(),
                    ).mean()
            with profiling_fn("backward"):
                accelerator.backward(discriminator_loss)
            with profiling_fn("gradient_sync_and_step"):
                for name in ("msd", "mpd"):
                    synchronize_gradients(accelerator, modules, (name,))
                    optimizer.step(name)
                    modules[name].requires_grad_(False)

        optimizer.zero_grad()
        with profiling_fn("generator_losses"):
            with accelerator.autocast():
                with profiling_fn("stft_loss"):
                    mel_loss = losses.stft(reconstructed, waveform)
                with profiling_fn("adversarial_loss"):
                    generator_loss = losses.generator(
                        waveform,
                        reconstructed,
                    ).mean()
                with profiling_fn("wavlm_loss"):
                    wavlm_loss = losses.wavlm(
                        waveform.detach().squeeze(1),
                        reconstructed.squeeze(1),
                    ).mean()
        with profiling_fn("generator_losses.duration_and_alignment"):
            duration_loss = torch.zeros((), device=batch.texts.device)
            cross_entropy_loss = torch.zeros((), device=batch.texts.device)
            items = zip(
                duration_predictions,
                duration_targets,
                batch.input_lengths,
            )
            for prediction, target, length in items:
                prediction = prediction[:length, :]
                target = target[:length].long()
                binary_target = torch.zeros_like(prediction)
                for position in range(binary_target.shape[0]):
                    binary_target[position, : target[position]] = 1
                predicted_duration = torch.sigmoid(prediction).sum(axis=1)
                duration_loss += F.l1_loss(
                    predicted_duration[1 : length - 1],
                    target[1 : length - 1],
                )
                cross_entropy_loss += F.binary_cross_entropy_with_logits(
                    prediction.flatten(),
                    binary_target.flatten(),
                )
            duration_loss /= batch.texts.size(0)
            cross_entropy_loss /= batch.texts.size(0)
            sequence_loss = torch.zeros((), device=batch.texts.device)
            items = zip(
                alignment_predictions,
                batch.texts,
                batch.input_lengths,
            )
            for prediction, target, length in items:
                sequence_loss += F.cross_entropy(
                    prediction[:length],
                    target[:length],
                )
            sequence_loss /= batch.texts.size(0)
            monotonic_loss = F.l1_loss(
                soft_alignment,
                monotonic_alignment,
            ) * 10
        with profiling_fn("generator_losses.total"):
            total_loss = (
                loss_config.lambda_mel * mel_loss
                + loss_config.lambda_F0 * f0_loss
                + loss_config.lambda_ce * cross_entropy_loss
                + loss_config.lambda_norm * norm_loss
                + loss_config.lambda_dur * duration_loss
                + loss_config.lambda_gen * generator_loss
                + loss_config.lambda_slm * wavlm_loss
                + loss_config.lambda_sty * style_loss
                + loss_config.lambda_diff * diffusion_loss
                + loss_config.lambda_mono * monotonic_loss
                + loss_config.lambda_s2s * sequence_loss
            )
        if not torch.isfinite(total_loss):
            for name in ("msd", "mpd"):
                modules[name].requires_grad_(True)
            self.skipped_steps += 1
            logger.warning(
                "skipping non-finite generator step=%s skipped_steps=%s",
                self.step,
                self.skipped_steps,
            )
            return {
                "step_skipped": 1.0,
                "skipped_steps": float(self.skipped_steps),
            }

        with profiling_fn("generator_backward"):
            accelerator.backward(total_loss)
        for name in ("msd", "mpd"):
            modules[name].requires_grad_(True)
        generator_modules = [
            "bert_encoder",
            "bert",
            "predictor",
            "predictor_encoder",
            "style_encoder",
            "decoder",
            "text_encoder",
            "text_aligner",
        ]
        if self.step >= loss_config.diffusion_start_step:
            generator_modules.append("diffusion")
        with profiling_fn("generator_gradient_sync"):
            synchronize_gradients(accelerator, modules, generator_modules)
        with profiling_fn("generator_optimizer_step"):
            for name in generator_modules:
                optimizer.step(name)

        joint_discriminator: torch.Tensor | float = 0.0
        joint_generator: torch.Tensor | float = 0.0
        if self.step >= loss_config.joint_start_step:
            use_individual = bool(np.random.rand() < 0.5)
            reference_texts = (
                batch.texts if use_individual else batch.reference_texts
            )
            reference_lengths = (
                batch.input_lengths
                if use_individual
                else batch.reference_lengths
            )
            with profiling_fn("joint"):
                with accelerator.autocast():
                    with profiling_fn("forward"):
                        joint_output = losses.slm_adversarial(
                            self.step,
                            waveform,
                            reference_prediction,
                            batch.waves,
                            batch.mel_lengths,
                            reference_texts,
                            reference_lengths,
                            use_individual,
                            style_target.detach(),
                            style_reference,
                        )
            if joint_output is not None:
                joint_discriminator, joint_generator, _ = joint_output
                optimizer.zero_grad()
                with profiling_fn("joint.generator_backward"):
                    accelerator.backward(joint_generator)
                with profiling_fn("joint.gradient_processing"):
                    squared_norms = {}
                    for name, module in modules.items():
                        squared_norms[name] = sum(
                            parameter.grad.detach().data.norm(2).item() ** 2
                            for parameter in module.parameters()
                            if parameter.grad is not None
                            and parameter.requires_grad
                        )
                    predictor_norm = squared_norms["predictor"] ** 0.5
                    if predictor_norm > self.config.slmadv_params.thresh:
                        for module in modules.values():
                            for parameter in module.parameters():
                                if parameter.grad is not None:
                                    parameter.grad *= 1 / predictor_norm
                    scaled_parameters = (
                        *predictor_core.duration_proj.parameters(),
                        *predictor_core.lstm.parameters(),
                        *modules.diffusion.parameters(),
                    )
                    for parameter in scaled_parameters:
                        if parameter.grad is not None:
                            parameter.grad *= self.config.slmadv_params.scale
                joint_modules = ("bert_encoder", "bert", "predictor", "diffusion")
                with profiling_fn("joint.gradient_sync"):
                    synchronize_gradients(accelerator, modules, joint_modules)
                with profiling_fn("joint.optimizer_step"):
                    for name in joint_modules:
                        optimizer.step(name)
                if isinstance(joint_discriminator, torch.Tensor):
                    optimizer.zero_grad()
                    with profiling_fn("joint.discriminator_backward"):
                        accelerator.backward(
                            joint_discriminator,
                            retain_graph=True,
                        )
                    with profiling_fn("joint.discriminator_sync_and_step"):
                        synchronize_gradients(accelerator, modules, ("wd",))
                        optimizer.step("wd")

        metrics = {
            "mel_loss": mel_loss,
            "gen_loss": generator_loss,
            "d_loss": discriminator_loss,
            "ce_loss": cross_entropy_loss,
            "dur_loss": duration_loss,
            "slm_loss": wavlm_loss,
            "norm_loss": norm_loss,
            "F0_loss": f0_loss,
            "sty_loss": style_loss,
            "diff_loss": diffusion_loss,
            "d_loss_slm": joint_discriminator,
            "gen_loss_slm": joint_generator,
            "s2s_loss": sequence_loss,
            "mono_loss": monotonic_loss,
            "step_skipped": 0.0,
            "skipped_steps": float(self.skipped_steps),
        }
        return metrics
