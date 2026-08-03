from dataclasses import dataclass

import torch
from torch import Tensor

from ...stages import (
    ProsodySource,
    StyleSource,
    TrainableModule,
    TrainingLoss,
    TrainingStageSpec,
)
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm
from .prosody_sampling import (
    prosody_inputs,
    sample_alpha_flow_features,
    sample_target_prosody_input,
)
from .rvq_initialization import initialize_rvq_codebooks


STYLE_PIPELINE_LOSSES = {
    TrainingLoss.ALPHA_FLOW,
    TrainingLoss.DURATION,
    TrainingLoss.DURATION_CE,
    TrainingLoss.F0,
    TrainingLoss.NORM,
    TrainingLoss.PROSODY_ADVERSARIAL,
    TrainingLoss.RVQ,
    TrainingLoss.STYLE_NUISANCE,
    TrainingLoss.XCOV,
}


@dataclass
class LaterStageOutput:
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


def later_stage_forward(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    stage: TrainingStageSpec,
    enabled: set[TrainingLoss],
    text_mask: Tensor,
    text_encoding: Tensor,
    soft_alignment: Tensor,
    monotonic: Tensor,
    duration_targets: Tensor,
    alpha_flow_step: int,
) -> LaterStageOutput:
    modules = runtime.models.modules
    device = batch.mels.device
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
    style_required = bool(enabled & STYLE_PIPELINE_LOSSES)
    bert_required = (
        duration_required
        or prosody_required
        or TrainingLoss.ALPHA_FLOW in enabled
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
        target_f0 = target_f0.squeeze(-1)
        with torch.no_grad():
            target_energy = log_norm(batch.mels.unsqueeze(1)).squeeze(1)
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
        TrainingLoss.RVQ in enabled
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
    if TrainingLoss.ALPHA_FLOW in enabled:
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
    if TrainingLoss.PROSODY_ADVERSARIAL in enabled:
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
            (position_features, duration_targets.unsqueeze(1)), dim=1
        )
        duration_fake = torch.cat(
            (position_features, predicted_duration.unsqueeze(1)), dim=1
        )
    return LaterStageOutput(
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
    )
