import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ...stages import (
    ProsodySource,
    StyleSource,
    TrainableModule,
    TrainingStageSpec,
)
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path
from .prosody_sampling import (
    prosody_inputs,
    sample_alpha_flow_features,
    sample_target_prosody_input,
)
from .rvq_initialization import initialize_rvq_codebooks
from .training_crops import crop_training_batch, sample_voice_prompts


@dataclass
class ForwardOutput:
    alignment_predictions: Tensor
    soft_alignment: Tensor
    monotonic_alignment: Tensor
    duration_targets: Tensor
    duration_predictions: Tensor
    prosody_real: Tensor
    prosody_fake: Tensor
    duration_real: Tensor
    duration_fake: Tensor
    target_f0: Tensor
    target_energy: Tensor
    predicted_f0: Tensor
    predicted_energy: Tensor
    style_target: Tensor
    rvq_loss: Tensor
    alpha_flow_loss: Tensor
    voice: Tensor
    reconstructed: Tensor
    waveform: Tensor


def model_forward(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    stage: TrainingStageSpec,
    alpha_flow_step: int,
    max_decoder_frames: int,
) -> ForwardOutput:
    modules = runtime.models.modules
    device = batch.mels.device
    weights = stage.loss_weights
    mask = length_to_mask(batch.mel_lengths // (2**runtime.models.n_down), device)
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
    soft_alignment = soft_alignment.masked_fill(~alignment_mask.bool(), 0.0)
    monotonic = maximum_path(soft_alignment, alignment_mask)
    duration_targets = monotonic.sum(-1).detach()
    text_encoding = modules.text_encoder(
        batch.texts,
        batch.input_lengths,
        text_mask,
    )
    duration_required = (
        weights.duration > 0
        or weights.duration_ce > 0
        or weights.prosody_adversarial > 0
    )
    prosody_required = (
        stage.prosody_source is ProsodySource.PREDICTED
        or weights.f0 > 0
        or weights.norm > 0
        or weights.prosody_adversarial > 0
    )
    style_required = (
        weights.alpha_flow > 0
        or weights.duration > 0
        or weights.duration_ce > 0
        or weights.f0 > 0
        or weights.norm > 0
        or weights.prosody_adversarial > 0
        or weights.rvq > 0
        or weights.style_nuisance > 0
        or weights.xcov > 0
    )
    bert_required = (
        duration_required
        or prosody_required
        or weights.alpha_flow > 0
    )
    bert = batch.mels.new_zeros((batch.texts.size(0), batch.texts.size(1), 1))
    if bert_required:
        bert = modules.bert(
            batch.texts,
            attention_mask=(~text_mask).int(),
            language_ids=batch.language_ids,
            modality_ids=batch.modality_ids,
        )

    target_f0 = batch.mels.new_zeros((batch.mels.size(0), batch.mels.size(-1)))
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
        weights.rvq > 0
        and TrainableModule.PROSODY_ENCODER in stage.trainable_modules
    ):
        encoder_inputs, encoder_lengths = sample_target_prosody_input(
            batch,
            style_inputs,
        )

    style_target = batch.mels.new_zeros((batch.mels.size(0), 512, 1))
    rvq_loss = batch.mels.new_zeros(())
    if style_required:
        with torch.autocast(device_type=device.type, enabled=False):
            continuous_style = modules.prosody_encoder(
                encoder_inputs.float(),
                encoder_lengths,
            )
            if stage.style_source is StyleSource.QUANTIZED:
                initialize_rvq_codebooks(
                    modules.quantizer,
                    continuous_style,
                    runtime.accelerator,
                )
                style_target, commitment, codebook, _ = modules.quantizer(
                    continuous_style
                )
                rvq_loss = commitment.mean() + codebook.mean()
            else:
                style_target = continuous_style

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
            style_target.detach(),
            bert,
            alpha_features,
            batch.input_lengths,
            alpha_flow_step,
        )

    duration_predictions = style_target.new_zeros((*duration_targets.shape, 1))
    predicted_f0 = target_f0.new_zeros(target_f0.shape)
    predicted_energy = target_energy.new_zeros(target_energy.shape)
    if duration_required or prosody_required:
        duration_encoding = modules.bert_encoder(bert).transpose(-1, -2)
        if duration_required:
            duration_predictions = modules.duration_predictor(
                duration_encoding,
                style_target,
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
    predicted_f0 = predicted_f0.masked_fill(full_mask, 0.0)
    predicted_energy = predicted_energy.masked_fill(full_mask, 0.0)

    prosody_fake = style_inputs.new_zeros(style_inputs.shape)
    duration_shape = (batch.texts.size(0), 513, monotonic.size(1))
    duration_real = batch.mels.new_zeros(duration_shape)
    duration_fake = batch.mels.new_zeros(duration_shape)
    if weights.prosody_adversarial > 0:
        prosody_fake = prosody_inputs(
            modules.position_embedding,
            monotonic,
            predicted_f0,
            predicted_energy,
        )
        positions = torch.arange(monotonic.size(1), device=device)
        position_features = modules.position_embedding(positions).transpose(0, 1)
        position_features = position_features.unsqueeze(0).expand(
            batch.texts.size(0), -1, -1
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
        max_decoder_frames // 2,
    )
    voice_dim = runtime.models.parameters.style_dim
    voice = batch.mels.new_zeros((batch.texts.size(0), voice_dim))
    decoder_voice = voice
    decoder_text = text_encoding
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
        prompt_mels = sample_voice_prompts(batch)
        with torch.autocast(device_type=device.type, enabled=False):
            decoder_voice, decoder_text = modules.voice_encoder(
                prompt_mels.float(),
                text_encoding.float(),
                batch.input_lengths,
                text_encoding.size(-1),
            )
            if random.random() < 0.2:
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
    crops = crop_training_batch(
        batch,
        aligned_text,
        target_f0,
        target_energy,
        predicted_f0,
        predicted_energy,
        crop_frames,
    )
    (
        aligned_crop,
        target_f0_crop,
        target_energy_crop,
        predicted_f0_crop,
        predicted_energy_crop,
        cropped_mels,
        waveform,
    ) = crops
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
            target_energy_crop = log_norm(cropped_mels.unsqueeze(1)).squeeze(1)
    if stage.prosody_source is ProsodySource.PREDICTED:
        decoder_f0 = predicted_f0_crop
        decoder_energy = predicted_energy_crop
    else:
        decoder_f0, decoder_energy = target_f0_crop, target_energy_crop
    reconstructed = waveform
    if weights.mel > 0:
        reconstructed = modules.decoder(
            aligned_crop,
            decoder_f0,
            decoder_energy,
            decoder_voice,
        )
    return ForwardOutput(
        alignment_predictions,
        soft_alignment,
        monotonic,
        duration_targets,
        duration_predictions,
        style_inputs,
        prosody_fake,
        duration_real,
        duration_fake,
        target_f0,
        target_energy,
        predicted_f0,
        predicted_energy,
        style_target,
        rvq_loss,
        alpha_loss,
        voice,
        reconstructed,
        waveform,
    )
