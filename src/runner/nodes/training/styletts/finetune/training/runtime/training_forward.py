from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path
from ...stages import (
    ProsodySource,
    StyleSource,
    TrainableModule,
    TrainingLoss,
    TrainingStageSpec,
)
from .training_crops import crop_training_batch, sample_voice_prompts
from .rvq_initialization import initialize_rvq_codebooks
from .stage_requirements import requires_voice
from .prosody_sampling import (
    prosody_inputs,
    sample_alpha_flow_features,
    sample_target_prosody_input,
)
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
    enabled = set(stage.enabled_losses)
    mask = length_to_mask(batch.mel_lengths // (2**runtime.models.n_down), device)
    text_mask = length_to_mask(batch.input_lengths, device)
    _, alignment_predictions, soft_alignment = modules.text_aligner(batch.mels, mask, batch.texts)
    soft_alignment = soft_alignment.transpose(-1, -2)[..., 1:].transpose(-1, -2)
    alignment_mask = mask_from_lens(
        soft_alignment,
        batch.input_lengths,
        batch.mel_lengths // (2**runtime.models.n_down),
    )
    monotonic = maximum_path(soft_alignment, alignment_mask)
    duration_targets = monotonic.sum(-1).detach()
    bert = modules.bert(
        batch.texts,
        attention_mask=(~text_mask).int(),
        language_ids=batch.language_ids,
        modality_ids=batch.modality_ids,
    )
    text_encoding = modules.text_encoder(
        batch.texts,
        batch.input_lengths,
        text_mask,
    )
    duration_required = bool(
        enabled
        & {
            TrainingLoss.DURATION,
            TrainingLoss.DURATION_CE,
            TrainingLoss.PROSODY_ADVERSARIAL,
        }
    )
    prosody_required = (
        stage.prosody_source is ProsodySource.PREDICTED
        or bool(
            enabled
            & {
                TrainingLoss.F0,
                TrainingLoss.NORM,
                TrainingLoss.PROSODY_ADVERSARIAL,
            }
        )
    )
    if TrainableModule.PITCH_EXTRACTOR in stage.trainable_modules:
        raw_f0, _, _ = modules.pitch_extractor(
            batch.mels.unsqueeze(1),
            batch.mel_lengths,
        )
    else:
        with torch.no_grad():
            raw_f0, _, _ = modules.pitch_extractor(
                batch.mels.unsqueeze(1),
                batch.mel_lengths,
            )
    raw_f0 = raw_f0.squeeze(-1)
    with torch.no_grad():
        raw_energy = log_norm(batch.mels.unsqueeze(1)).squeeze(1)
    full_mask = length_to_mask(batch.mel_lengths, device)
    raw_f0 = raw_f0.masked_fill(full_mask, 0.0)
    raw_energy.masked_fill_(full_mask, 0.0)
    style_inputs = prosody_inputs(
        modules.position_embedding,
        monotonic,
        raw_f0,
        raw_energy,
    )
    encoder_inputs = style_inputs
    encoder_lengths = batch.mel_lengths.to(device) // 2
    if (
        TrainingLoss.RVQ in enabled
        and TrainableModule.PROSODY_ENCODER in stage.trainable_modules
    ):
        encoder_inputs, encoder_lengths = sample_target_prosody_input(
            batch,
            style_inputs,
        )
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
            rvq_loss = continuous_style.new_zeros(())
    alpha_loss = style_target.new_zeros(())
    if TrainingLoss.ALPHA_FLOW in enabled:
        alpha_features = sample_alpha_flow_features(
            batch,
            text_encoding,
            soft_alignment,
            monotonic,
            raw_f0,
            raw_energy,
        )
        alpha_loss = modules.alpha_flow(
            style_target.detach(),
            bert,
            alpha_features,
            batch.input_lengths,
            alpha_flow_step,
        )
    duration_predictions = style_target.new_zeros(
        (*duration_targets.shape, 1)
    )
    predicted_f0 = raw_f0.new_zeros(raw_f0.shape)
    predicted_energy = raw_energy.new_zeros(raw_energy.shape)
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
    full_mask = length_to_mask(batch.mel_lengths.to(device), device)
    predicted_f0_masked = predicted_f0.masked_fill(full_mask, 0.0)
    predicted_energy_masked = predicted_energy.masked_fill(full_mask, 0.0)
    prosody_fake = prosody_inputs(
        modules.position_embedding,
        monotonic,
        predicted_f0_masked,
        predicted_energy_masked,
    )
    positions = torch.arange(monotonic.size(1), device=device)
    position_features = modules.position_embedding(positions).transpose(0, 1)
    position_features = position_features.unsqueeze(0).expand(batch.texts.size(0), -1, -1)
    predicted_duration = torch.sigmoid(duration_predictions).sum(-1).masked_fill(text_mask, 0.0)
    duration_real = torch.cat((position_features, duration_targets.unsqueeze(1)), dim=1)
    duration_fake = torch.cat((position_features, predicted_duration.unsqueeze(1)), dim=1)
    crop_frames = min(
        int(batch.mel_lengths.min().item() / 2 - 1),
        max_decoder_frames // 2,
    )
    voice_dim = runtime.models.parameters.style_dim
    voice = batch.mels.new_zeros((batch.texts.size(0), voice_dim))
    decoder_voice = voice
    decoder_text = text_encoding
    if requires_voice(enabled):
        prompt_mels = sample_voice_prompts(batch)
        with torch.autocast(device_type=device.type, enabled=False):
            decoder_voice, decoder_text = modules.voice_encoder(
                prompt_mels.float(),
                text_encoding.float(),
                batch.input_lengths,
                text_encoding.size(-1),
            )
            voice = F.normalize(decoder_voice, dim=-1)
    aligned_text = decoder_text @ monotonic
    crops = crop_training_batch(
        batch,
        aligned_text,
        raw_f0,
        raw_energy,
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
        waveform,
    ) = crops
    if stage.prosody_source is ProsodySource.PREDICTED:
        decoder_f0 = predicted_f0_crop
        decoder_energy = predicted_energy_crop
    else:
        decoder_f0, decoder_energy = target_f0_crop, target_energy_crop
    reconstructed = waveform
    if TrainingLoss.MEL in stage.enabled_losses:
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
        raw_f0,
        raw_energy,
        predicted_f0_masked,
        predicted_energy_masked,
        style_target,
        rvq_loss,
        alpha_loss,
        voice,
        reconstructed,
        waveform,
    )
