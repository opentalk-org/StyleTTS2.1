import logging
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import TrainingConfig
from ..data import TrainingBatch
from ..gradient_sync import synchronize_gradients
from ..losses import (
    prosody_discriminator_loss,
    prosody_generator_losses,
    slm_discriminator_loss as compute_slm_discriminator_loss,
    slm_generator_loss,
    speaker_losses,
    reconstruction_loss,
    waveform_discriminator_losses,
    waveform_generator_losses,
    wavlm_feature_loss,
)
from ..profiling import profiling_fn, set_profiling_step
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path
from ...stages import (
    ProsodySource,
    StyleSource,
    TrainableModule,
    TrainingLoss,
    TrainingStageSpec,
    stage_for_step,
)
from .batch_ops import (
    crop_training_batch,
    prosody_inputs,
    sample_alpha_flow_features,
    sample_target_prosody_input,
    sample_voice_prompts,
)
from .gradient_norms import gradient_norm_metrics


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
        self.step = runtime.initial_step

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
        if weights.mel > 0:
            training_modules.update(("pitch_extractor", "text_aligner"))
        self.runtime.models.set_training_mode(training_modules)

    def train_step(self, batch: TrainingBatch) -> dict[str, torch.Tensor | float]:
        set_profiling_step(self.step)
        runtime = self.runtime
        accelerator = runtime.accelerator
        modules = runtime.models.modules
        optimizer = runtime.optimizer
        batch = batch.to(accelerator.device)
        stage = stage_for_step(self.config.training_stages, self.step)
        weights = stage.loss_weights
        trainable = tuple(module.value for module in stage.trainable_modules)
        measure_gradient_norms = (
            (self.step + 1) % self.config.log_every_steps == 0
        )

        for name, module in modules.items():
            module.requires_grad_(name in trainable)
        waveform_adversarial = weights.adversarial > 0
        modules.mpd.requires_grad_(waveform_adversarial)
        modules.msd.requires_grad_(waveform_adversarial)
        prosody_adversarial = weights.prosody_adversarial > 0
        modules.prosody_discriminator.requires_grad_(prosody_adversarial)
        modules.duration_discriminator.requires_grad_(prosody_adversarial)
        slm_adversarial = weights.slm_adversarial > 0
        modules.wd.requires_grad_(slm_adversarial)
        gradient_metrics: dict[str, torch.Tensor | float] = {}
        voice_required = (
            waveform_adversarial
            or weights.mel > 0
            or slm_adversarial
            or weights.speaker_feature > 0
            or weights.speaker_similarity > 0
            or weights.style_nuisance > 0
            or weights.wavlm > 0
            or weights.xcov > 0
        )

        with accelerator.autocast():
            alpha_flow_step = max(0, self.step - self.alpha_flow_start)
            device = batch.mels.device
            mask = length_to_mask(
                batch.mel_lengths // (2**runtime.models.n_down),
                device,
            )
            text_mask = length_to_mask(batch.input_lengths, device)
            _, alignment_predictions, soft_alignment = modules.text_aligner(
                batch.mels,
                mask,
                batch.texts,
            )
            soft_alignment = soft_alignment.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            alignment_mask = mask_from_lens(
                soft_alignment,
                batch.input_lengths,
                batch.mel_lengths // (2**runtime.models.n_down),
            )
            soft_alignment = soft_alignment.masked_fill(
                ~alignment_mask.bool(),
                0.0,
            )
            monotonic = maximum_path(soft_alignment, alignment_mask)
            duration_targets = monotonic.sum(-1).detach()
            text_encoding = modules.text_encoder(
                batch.texts,
                batch.input_lengths,
                text_mask,
                batch.language_ids,
            )
            duration_required = (
                weights.duration > 0
                or weights.duration_ce > 0
                or prosody_adversarial
            )
            prosody_required = (
                stage.prosody_source is ProsodySource.PREDICTED
                or weights.f0 > 0
                or weights.norm > 0
                or prosody_adversarial
            )
            style_required = (
                weights.alpha_flow > 0
                or weights.duration > 0
                or weights.duration_ce > 0
                or weights.f0 > 0
                or weights.norm > 0
                or prosody_adversarial
                or weights.style_nuisance > 0
                or weights.xcov > 0
            )
            bert_required = duration_required or prosody_required or weights.alpha_flow > 0
            bert = batch.mels.new_zeros(
                (batch.texts.size(0), batch.texts.size(1), 1)
            )
            if bert_required:
                bert = modules.bert(
                    batch.texts,
                    attention_mask=(~text_mask).int(),
                    language_ids=batch.language_ids,
                    modality_ids=batch.modality_ids,
                )

            target_f0 = batch.mels.new_zeros(
                (batch.mels.size(0), batch.mels.size(-1))
            )
            target_energy = target_f0.clone()
            if style_required:
                with torch.no_grad():
                    target_f0, _, _ = modules.pitch_extractor(
                        batch.mels.unsqueeze(1),
                        batch.mel_lengths,
                    )
                    target_energy = log_norm(batch.mels.unsqueeze(1)).squeeze(1)
                target_f0 = target_f0.squeeze(-1)
            full_mask = length_to_mask(batch.mel_lengths, device)
            target_f0 = target_f0.masked_fill(full_mask, 0.0)
            target_energy.masked_fill_(full_mask, 0.0)

            style_inputs = batch.mels.new_zeros(
                (batch.mels.size(0), 514, monotonic.size(-1))
            )
            if style_required:
                style_inputs = prosody_inputs(
                    modules.position_embedding,
                    monotonic,
                    target_f0,
                    target_energy,
                )
            encoder_inputs = style_inputs
            encoder_lengths = batch.mel_lengths.to(device) // 2
            if (
                stage.style_source is StyleSource.QUANTIZED
                and TrainableModule.PROSODY_ENCODER in stage.trainable_modules
            ):
                encoder_inputs, encoder_lengths = sample_target_prosody_input(
                    batch,
                    style_inputs,
                )

            style_target = batch.mels.new_zeros((batch.mels.size(0), 512, 1))
            continuous_decode_style = style_target
            continuous_latent = batch.mels.new_zeros(
                (
                    batch.mels.size(0),
                    modules.quantizer.latent_dim,
                    1,
                )
            )
            quantization_error = batch.mels.new_zeros(())
            dual_decode = False
            if style_required:
                with torch.autocast(device_type=device.type, enabled=False):
                    encoded_style = modules.prosody_encoder(
                        encoder_inputs.float(),
                        encoder_lengths,
                    )
                    if stage.style_source is StyleSource.QUANTIZED:
                        latents = modules.quantizer(encoded_style)
                        continuous_latent = latents.continuous
                        continuous_decode_style = latents.continuous_style
                        style_target = latents.quantized_style
                        quantization_error = latents.quantization_error
                        dual_decode = (
                            TrainableModule.QUANTIZER in stage.trainable_modules
                        )
                    else:
                        style_target = encoded_style

            alpha_loss = style_target.new_zeros(())
            if weights.alpha_flow > 0:
                alpha_features = sample_alpha_flow_features(
                    batch,
                    text_encoding,
                    soft_alignment,
                    monotonic,
                    target_f0,
                    target_energy,
                )
                alpha_loss = modules.alpha_flow(
                    continuous_latent.detach(),
                    bert,
                    alpha_features,
                    batch.input_lengths,
                    alpha_flow_step,
                )

            duration_predictions = style_target.new_zeros(
                (*duration_targets.shape, 1)
            )
            predicted_f0 = target_f0.new_zeros(target_f0.shape)
            predicted_energy = target_energy.new_zeros(target_energy.shape)
            continuous_duration_predictions = duration_predictions
            continuous_predicted_f0 = predicted_f0
            continuous_predicted_energy = predicted_energy
            if duration_required or prosody_required:
                duration_encoding = modules.bert_encoder(bert).transpose(-1, -2)
                if duration_required:
                    duration_predictions = modules.duration_predictor(
                        duration_encoding,
                        style_target,
                        batch.input_lengths,
                        duration_encoding.size(-1),
                    )
                    if dual_decode:
                        continuous_duration_predictions = modules.duration_predictor(
                            duration_encoding,
                            continuous_decode_style,
                            batch.input_lengths,
                            duration_encoding.size(-1),
                        )
                if prosody_required:
                    aligned_duration = duration_encoding @ monotonic
                    predicted_f0, predicted_energy = modules.prosody_predictor(
                        aligned_duration,
                        style_target,
                        batch.mel_lengths.to(device) // 2,
                        monotonic.size(-1),
                    )
                    if dual_decode:
                        (
                            continuous_predicted_f0,
                            continuous_predicted_energy,
                        ) = modules.prosody_predictor(
                            aligned_duration,
                            continuous_decode_style,
                            batch.mel_lengths.to(device) // 2,
                            monotonic.size(-1),
                        )
            predicted_f0 = predicted_f0.masked_fill(full_mask, 0.0)
            predicted_energy = predicted_energy.masked_fill(full_mask, 0.0)
            continuous_predicted_f0 = continuous_predicted_f0.masked_fill(
                full_mask,
                0.0,
            )
            continuous_predicted_energy = continuous_predicted_energy.masked_fill(
                full_mask,
                0.0,
            )

            prosody_fake = style_inputs.new_zeros(style_inputs.shape)
            duration_shape = (batch.texts.size(0), 513, monotonic.size(1))
            duration_real = batch.mels.new_zeros(duration_shape)
            duration_fake = batch.mels.new_zeros(duration_shape)
            if prosody_adversarial:
                prosody_fake = prosody_inputs(
                    modules.position_embedding,
                    monotonic,
                    predicted_f0,
                    predicted_energy,
                )
                positions = torch.arange(monotonic.size(1), device=device)
                position_features = modules.position_embedding(positions).transpose(0, 1)
                position_features = position_features.unsqueeze(0).expand(
                    batch.texts.size(0),
                    -1,
                    -1,
                )
                predicted_duration = torch.sigmoid(duration_predictions).sum(-1)
                predicted_duration = predicted_duration.masked_fill(text_mask, 0.0)
                duration_real = torch.cat(
                    (position_features, duration_targets.unsqueeze(1)),
                    dim=1,
                )
                duration_fake = torch.cat(
                    (position_features, predicted_duration.unsqueeze(1)),
                    dim=1,
                )

            crop_frames = min(
                int(batch.mel_lengths.min().item() / 2 - 1),
                int(
                    stage.max_decoder_seconds
                    * self.config.preprocess_params.sr
                    / self.config.preprocess_params.spect_params.hop_length
                    / 2
                ),
            )
            voice_dim = runtime.models.parameters.style_dim
            voice = batch.mels.new_zeros((batch.texts.size(0), voice_dim))
            decoder_voice = voice
            decoder_text = text_encoding
            if voice_required:
                prompt_mels = sample_voice_prompts(batch)
                with torch.autocast(device_type=device.type, enabled=False):
                    decoder_voice, decoder_text = modules.voice_encoder(
                        prompt_mels.float(),
                        text_encoding.float(),
                        batch.input_lengths,
                        text_encoding.size(-1),
                    )
                    if random.random() < stage.voice_conditioning_dropout:
                        with torch.no_grad():
                            null_voice, null_text = modules.voice_encoder(
                                torch.zeros_like(prompt_mels).float(),
                                decoder_text,
                                batch.input_lengths,
                                decoder_text.size(-1),
                            )
                        if bool(random.getrandbits(1)):
                            decoder_voice = null_voice
                        else:
                            decoder_text = null_text
                    voice = F.normalize(decoder_voice, dim=-1)
            decoder_alignment = (
                soft_alignment if bool(random.getrandbits(1)) else monotonic
            )
            aligned_text = decoder_text @ decoder_alignment
            (
                aligned_crop,
                target_f0_crop,
                target_energy_crop,
                predicted_f0_crop,
                predicted_energy_crop,
                cropped_mels,
                waveform,
            ) = crop_training_batch(
                batch,
                aligned_text,
                target_f0,
                target_energy,
                predicted_f0,
                predicted_energy,
                crop_frames,
            )
            if weights.mel > 0:
                cropped_lengths = batch.mel_lengths.new_full(
                    (cropped_mels.size(0),),
                    cropped_mels.size(-1),
                )
                with torch.no_grad():
                    target_f0_crop, _, _ = modules.pitch_extractor(
                        cropped_mels.unsqueeze(1),
                        cropped_lengths,
                    )
                target_f0_crop = target_f0_crop.squeeze(-1)
                with torch.no_grad():
                    target_energy_crop = log_norm(
                        cropped_mels.unsqueeze(1)
                    ).squeeze(1)
            if stage.prosody_source is ProsodySource.PREDICTED:
                decoder_f0 = predicted_f0_crop
                decoder_energy = predicted_energy_crop
            else:
                decoder_f0 = target_f0_crop
                decoder_energy = target_energy_crop
            reconstructed = waveform
            if weights.mel > 0:
                reconstructed = modules.decoder(
                    aligned_crop,
                    decoder_f0,
                    decoder_energy,
                    decoder_voice,
                )
        gan_metrics: dict[str, torch.Tensor | float] = {}
        prosody_discriminator_total = reconstructed.new_zeros(())
        slm_discriminator_loss = reconstructed.new_zeros(())

        if waveform_adversarial:
            for name, discriminator in (
                ("mpd", modules.mpd),
                ("msd", modules.msd),
            ):
                optimizer.zero_grad(name)
                with accelerator.autocast():
                    real_scores, generated_scores, _, _ = discriminator(
                        waveform.detach().float(),
                        reconstructed.detach().float(),
                        return_features=False,
                    )
                    discriminator_losses = waveform_discriminator_losses(
                        real_scores,
                        generated_scores,
                    )
                    prefix = f"gan/{name}/discriminator"
                    gan_metrics.update(
                        {
                            f"{prefix}/real_lsgan": (
                                discriminator_losses.real_lsgan.detach()
                            ),
                            f"{prefix}/generated_lsgan": (
                                discriminator_losses.generated_lsgan.detach()
                            ),
                            f"{prefix}/tprls": (
                                discriminator_losses.tprls.detach()
                            ),
                            f"{prefix}/real_accuracy": (
                                discriminator_losses.real_accuracy.detach()
                            ),
                            f"{prefix}/generated_accuracy": (
                                discriminator_losses.generated_accuracy.detach()
                            ),
                            f"{prefix}/accuracy": (
                                discriminator_losses.accuracy.detach()
                            ),
                        }
                    )
                accelerator.backward(discriminator_losses.total)
                synchronize_gradients(accelerator, modules, (name,))
                if measure_gradient_norms:
                    gradient_metrics.update(
                        gradient_norm_metrics(accelerator, modules, (name,))
                    )
                optimizer.step(name)
                discriminator.requires_grad_(False)

        if prosody_adversarial:
            prosody_items = (
                (
                    "prosody_discriminator",
                    modules.prosody_discriminator,
                    prosody_fake,
                    style_inputs,
                    batch.mel_lengths.to(reconstructed.device) // 2,
                ),
                (
                    "duration_discriminator",
                    modules.duration_discriminator,
                    duration_fake,
                    duration_real,
                    batch.input_lengths.to(reconstructed.device),
                ),
            )
            for name, discriminator, fake, real, lengths in prosody_items:
                optimizer.zero_grad(name)
                generated_scores, _ = discriminator(
                    fake.detach().float(),
                    style_target.detach().float(),
                    lengths,
                    real.size(-1),
                )
                real_scores, _ = discriminator(
                    real.detach().float(),
                    style_target.detach().float(),
                    lengths,
                    real.size(-1),
                )
                loss = prosody_discriminator_loss(
                    real_scores,
                    generated_scores,
                    lengths,
                )
                accelerator.backward(loss)
                synchronize_gradients(accelerator, modules, (name,))
                if measure_gradient_norms:
                    gradient_metrics.update(
                        gradient_norm_metrics(accelerator, modules, (name,))
                    )
                optimizer.step(name)
                modules[name].requires_grad_(False)
                prosody_discriminator_total += loss.detach()

        if slm_adversarial:
            optimizer.zero_grad("wd")
            with accelerator.autocast():
                with torch.no_grad():
                    real_features = self.runtime.features.wavlm(
                        waveform.detach().squeeze(1)
                    )
                    generated_features = self.runtime.features.wavlm(
                        reconstructed.detach().squeeze(1)
                    )
                    real_input = self.runtime.features.wavlm.discriminator_input(
                        real_features
                    )
                    generated_input = self.runtime.features.wavlm.discriminator_input(
                        generated_features
                    )
                real_scores = modules.wd(real_input)
                generated_scores = modules.wd(generated_input)
                slm_discriminator_loss = compute_slm_discriminator_loss(
                    real_scores,
                    generated_scores,
                )
            accelerator.backward(slm_discriminator_loss)
            synchronize_gradients(accelerator, modules, ("wd",))
            if measure_gradient_norms:
                gradient_metrics.update(
                    gradient_norm_metrics(accelerator, modules, ("wd",))
                )
            optimizer.step("wd")
            modules.wd.requires_grad_(False)
            slm_discriminator_loss = slm_discriminator_loss.detach()

        for name in trainable:
            optimizer.zero_grad(name)
        dual_metrics: dict[str, torch.Tensor] = {}
        if style_required and stage.style_source is StyleSource.QUANTIZED:
            dual_metrics["rfsq_quantization_error"] = (
                quantization_error.detach()
            )
        with accelerator.autocast():
            zero = reconstructed.new_zeros(())
            losses = {item.value: zero for item in TrainingLoss}
            if weights.mel > 0:
                losses["mel"] = self.runtime.losses.stft(
                    reconstructed,
                    waveform,
                )
            if weights.f0 > 0:
                quantized_f0 = reconstruction_loss(
                    target_f0,
                    predicted_f0,
                    batch.mel_lengths,
                    divisor=10,
                )
                losses["f0"] = quantized_f0
                if dual_decode:
                    continuous_f0 = reconstruction_loss(
                        target_f0,
                        continuous_predicted_f0,
                        batch.mel_lengths,
                        divisor=10,
                    )
                    losses["f0"] = (quantized_f0 + continuous_f0) / 2
                    dual_metrics["f0_quantized"] = quantized_f0.detach()
                    dual_metrics["f0_continuous"] = continuous_f0.detach()
            if weights.norm > 0:
                quantized_norm = reconstruction_loss(
                    target_energy,
                    predicted_energy,
                    batch.mel_lengths,
                )
                losses["norm"] = quantized_norm
                if dual_decode:
                    continuous_norm = reconstruction_loss(
                        target_energy,
                        continuous_predicted_energy,
                        batch.mel_lengths,
                    )
                    losses["norm"] = (quantized_norm + continuous_norm) / 2
                    dual_metrics["norm_quantized"] = quantized_norm.detach()
                    dual_metrics["norm_continuous"] = continuous_norm.detach()
            if weights.duration > 0 or weights.duration_ce > 0:
                duration, duration_ce = self._duration_losses(
                    reconstructed,
                    duration_predictions,
                    duration_targets,
                    batch,
                )
                if dual_decode:
                    continuous_duration, continuous_duration_ce = (
                        self._duration_losses(
                            reconstructed,
                            continuous_duration_predictions,
                            duration_targets,
                            batch,
                        )
                    )
                    dual_metrics["duration_quantized"] = duration.detach()
                    dual_metrics["duration_continuous"] = (
                        continuous_duration.detach()
                    )
                    dual_metrics["duration_ce_quantized"] = duration_ce.detach()
                    dual_metrics["duration_ce_continuous"] = (
                        continuous_duration_ce.detach()
                    )
                    duration = (duration + continuous_duration) / 2
                    duration_ce = (duration_ce + continuous_duration_ce) / 2
                losses["duration"] = duration
                losses["duration_ce"] = duration_ce
            if (
                weights.sequence_alignment > 0
                or weights.monotonic_alignment > 0
            ):
                sequence, monotonic_loss = self._alignment_losses(
                    reconstructed,
                    alignment_predictions,
                    soft_alignment,
                    monotonic,
                    batch,
                )
                losses["sequence_alignment"] = sequence
                losses["monotonic_alignment"] = monotonic_loss
            if weights.alpha_flow > 0:
                losses["alpha_flow"] = alpha_loss
            if waveform_adversarial:
                period = waveform_generator_losses(
                    *modules.mpd(
                        waveform.float(),
                        reconstructed.float(),
                    )
                )
                scale = waveform_generator_losses(
                    *modules.msd(
                        waveform.float(),
                        reconstructed.float(),
                    )
                )
                losses["adversarial"] = period.total + scale.total
                gan_metrics.update(
                    {
                        "gan/mpd/generator/feature_matching": (
                            period.feature_matching.detach()
                        ),
                        "gan/mpd/generator/lsgan": period.lsgan.detach(),
                        "gan/mpd/generator/tprls": period.tprls.detach(),
                        "gan/msd/generator/feature_matching": (
                            scale.feature_matching.detach()
                        ),
                        "gan/msd/generator/lsgan": scale.lsgan.detach(),
                        "gan/msd/generator/tprls": scale.tprls.detach(),
                    }
                )
            if weights.wavlm > 0 or slm_adversarial:
                real_features = None
                if weights.wavlm > 0:
                    with torch.no_grad():
                        real_features = self.runtime.features.wavlm(
                            waveform.detach().squeeze(1)
                        )
                generated_features = self.runtime.features.wavlm(
                    reconstructed.squeeze(1)
                )
                if real_features is not None:
                    losses["wavlm"] = wavlm_feature_loss(
                        real_features,
                        generated_features,
                    )
                if slm_adversarial:
                    generated_input = (
                        self.runtime.features.wavlm.discriminator_input(
                            generated_features
                        )
                    )
                    losses["slm_adversarial"] = slm_generator_loss(
                        modules.wd(generated_input)
                    )
            if weights.speaker_feature > 0 or weights.speaker_similarity > 0:
                speaker = self.runtime.features.speaker
                if speaker is None:
                    raise RuntimeError("speaker features were not initialized")
                with torch.no_grad():
                    real_values, real_embedding = speaker(waveform.detach())
                generated_values, generated_embedding = (
                    speaker(reconstructed)
                )
                speaker_feature, speaker_similarity = speaker_losses(
                    real_values,
                    generated_values,
                    real_embedding,
                    generated_embedding,
                )
                losses["speaker_feature"] = speaker_feature
                losses["speaker_similarity"] = speaker_similarity
            if prosody_adversarial:
                prosody_lengths = batch.mel_lengths.to(reconstructed.device) // 2
                prosody_scores, prosody_fake_features = (
                    modules.prosody_discriminator(
                        prosody_fake.float(),
                        style_target.detach().float(),
                        prosody_lengths,
                        style_inputs.size(-1),
                    )
                )
                with torch.no_grad():
                    prosody_real_scores, prosody_real_features = (
                        modules.prosody_discriminator(
                            style_inputs.detach().float(),
                            style_target.detach().float(),
                            prosody_lengths,
                            style_inputs.size(-1),
                        )
                    )
                prosody, prosody_features = prosody_generator_losses(
                    prosody_real_scores,
                    prosody_scores,
                    prosody_real_features,
                    prosody_fake_features,
                    prosody_lengths,
                )
                duration_lengths = batch.input_lengths.to(reconstructed.device)
                duration_scores, duration_fake_features = (
                    modules.duration_discriminator(
                        duration_fake.float(),
                        style_target.detach().float(),
                        duration_lengths,
                        duration_real.size(-1),
                    )
                )
                with torch.no_grad():
                    duration_real_scores, duration_real_features = (
                        modules.duration_discriminator(
                            duration_real.detach().float(),
                            style_target.detach().float(),
                            duration_lengths,
                            duration_real.size(-1),
                        )
                    )
                duration, duration_features = prosody_generator_losses(
                    duration_real_scores,
                    duration_scores,
                    duration_real_features,
                    duration_fake_features,
                    duration_lengths,
                )
                adversarial = prosody + duration
                feature_matching = prosody_features + duration_features
                losses["prosody_generator_adversarial"] = adversarial
                losses["prosody_feature_matching"] = feature_matching
                losses["prosody_adversarial"] = adversarial + feature_matching
            if weights.style_nuisance > 0:
                positions = torch.arange(batch.texts.size(1), device=batch.texts.device)
                valid = positions[None, :] < batch.input_lengths.to(batch.texts.device)[:, None]
                tokens = F.one_hot(
                    batch.texts,
                    num_classes=self.runtime.models.parameters.n_token,
                ).to(torch.float32)
                content_bag = (tokens * valid.unsqueeze(-1)).amax(1)
                losses["style_nuisance"] = modules.factorization.style_nuisance_loss(
                    style_target,
                    batch.speaker_ids,
                    batch.language_ids,
                    content_bag,
                    min(1.0, self.step / 1000),
                )
            if weights.xcov > 0:
                losses["xcov"] = modules.factorization.cross_covariance(
                    voice,
                    style_target,
                )
        style_batch_std = style_target.mean(-1).std(
            0,
            unbiased=False,
        ).mean()
        total = self._weighted_total(losses, weights)
        with profiling_fn("loss_finite_check"):
            finite = bool(torch.isfinite(total).item())
        if finite:
            with profiling_fn("generator_backward"):
                accelerator.backward(total)
        else:
            self.skipped_steps += 1
            logger.warning("non-finite generator loss at step=%s", self.step)
            for name in trainable:
                for parameter in modules[name].parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        with profiling_fn("generator_gradient_sync"):
            synchronize_gradients(accelerator, modules, trainable)
        with profiling_fn("generator_gradient_metrics"):
            if measure_gradient_norms:
                gradient_metrics.update(
                    gradient_norm_metrics(
                        accelerator,
                        modules,
                        trainable,
                        group_name="generator",
                    )
                )
        with profiling_fn("generator_optimizer_step"):
            for name in trainable:
                optimizer.step(name)
        metrics = self._reported_metrics(
            voice,
            style_target,
            losses,
            weights,
            total,
            prosody_discriminator_total,
            slm_discriminator_loss,
            style_batch_std,
            finite,
            gradient_metrics,
            gan_metrics,
        )
        metrics.update(dual_metrics)
        if style_required and stage.style_source is StyleSource.QUANTIZED:
            metrics["continuous_latent_batch_std"] = (
                continuous_latent.mean(-1).std(0, unbiased=False).mean()
            )
        return metrics

    def _reported_metrics(
        self,
        voice: Tensor,
        style_target: Tensor,
        losses: dict[str, torch.Tensor],
        weights,
        total: torch.Tensor,
        prosody_discriminator_loss: torch.Tensor,
        slm_discriminator_loss: torch.Tensor,
        style_batch_std: torch.Tensor,
        finite: bool,
        gradient_metrics: dict[str, torch.Tensor | float],
        gan_metrics: dict[str, torch.Tensor | float],
    ) -> dict[str, torch.Tensor | float]:
        metrics: dict[str, torch.Tensor | float] = {
            item.value: losses[item.value]
            for item in TrainingLoss
            if getattr(weights, item.value) > 0
            and item is not TrainingLoss.ADVERSARIAL
        }
        metrics["total"] = total.detach()
        if weights.adversarial > 0:
            metrics.update(gan_metrics)
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
            metrics["voice_batch_std"] = voice.std(
                0,
                unbiased=False,
            ).mean()
        if (
            weights.alpha_flow > 0
            or weights.duration > 0
            or weights.duration_ce > 0
            or weights.f0 > 0
            or weights.norm > 0
            or weights.prosody_adversarial > 0
            or weights.style_nuisance > 0
            or weights.xcov > 0
        ):
            metrics["style_batch_std"] = style_batch_std
        if weights.style_nuisance > 0 or weights.xcov > 0:
            with torch.no_grad():
                projected = self.runtime.models.modules.factorization.style_projection(
                    style_target.mean(-1)
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
        metrics.update(gradient_metrics)
        return metrics

    @staticmethod
    def _duration_losses(
        reconstructed: Tensor,
        duration_predictions: Tensor,
        duration_targets: Tensor,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        duration = reconstructed.new_zeros(())
        cross_entropy = reconstructed.new_zeros(())
        items = zip(
            duration_predictions,
            duration_targets,
            batch.input_lengths,
            strict=True,
        )
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
    def _alignment_losses(
        reconstructed: Tensor,
        alignment_predictions: Tensor,
        soft_alignment: Tensor,
        monotonic_alignment: Tensor,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = reconstructed.new_zeros(())
        for prediction, target, length in zip(
            alignment_predictions,
            batch.texts,
            batch.input_lengths,
            strict=True,
        ):
            sequence = sequence + F.cross_entropy(
                prediction[:length],
                target[:length],
            )
        sequence = sequence / batch.texts.size(0)
        monotonic = F.l1_loss(soft_alignment, monotonic_alignment)
        return sequence, monotonic

    @staticmethod
    def _weighted_total(losses, weights) -> torch.Tensor:
        total = losses["mel"].new_zeros(())
        for item in TrainingLoss:
            total = total + losses[item.value] * getattr(weights, item.value)
        return total
