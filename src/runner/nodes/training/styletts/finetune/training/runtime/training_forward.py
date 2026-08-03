import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ...stages import (
    ProsodySource,
    TrainingLoss,
    TrainingStageSpec,
)
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import length_to_mask, log_norm, mask_from_lens, maximum_path
from .later_stage_forward import later_stage_forward
from .stage_requirements import requires_voice
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
    enabled = set(stage.enabled_losses)
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
    later = later_stage_forward(
        runtime,
        batch,
        stage,
        enabled,
        text_mask,
        text_encoding,
        soft_alignment,
        monotonic,
        duration_targets,
        alpha_flow_step,
    )
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
        later.target_f0,
        later.target_energy,
        later.predicted_f0,
        later.predicted_energy,
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
    if TrainingLoss.MEL in enabled:
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
        later.duration_predictions,
        later.prosody_real,
        later.prosody_fake,
        later.duration_real,
        later.duration_fake,
        later.target_f0,
        later.target_energy,
        later.predicted_f0,
        later.predicted_energy,
        later.style_target,
        later.rvq_loss,
        later.alpha_flow_loss,
        voice,
        reconstructed,
        waveform,
    )
