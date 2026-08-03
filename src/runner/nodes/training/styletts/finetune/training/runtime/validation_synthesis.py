import torch

from ...stages import (
    ProsodySource,
    StyleSource,
    TrainingStageSpec,
    ValidationDurationSource,
)
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from .prosody_sampling import prosody_inputs
from .validation_batch import predicted_alignment, resize_prosody


def synthesize_validation(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    stage: TrainingStageSpec,
    text_encoding: torch.Tensor,
    duration_encoding: torch.Tensor,
    bert: torch.Tensor,
    monotonic: torch.Tensor,
    target_f0: torch.Tensor,
    target_energy: torch.Tensor,
    alpha_flow_noise: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    modules = runtime.models.modules
    device = batch.mels.device
    conditioning = prosody_inputs(
        modules.position_embedding,
        monotonic,
        target_f0,
        target_energy,
    )
    with torch.autocast(device_type=device.type, enabled=False):
        continuous = modules.prosody_encoder(
            conditioning.float(),
            batch.mel_lengths.to(device) // 2,
        )
        style = continuous
        if stage.style_source is StyleSource.QUANTIZED:
            style, _, _, _ = modules.quantizer(continuous)
    if stage.validation.alpha_flow:
        style = modules.alpha_flow.sample(
            bert,
            conditioning,
            batch.input_lengths,
            noise=alpha_flow_noise,
        )
    duration_predictions = modules.duration_predictor(
        duration_encoding,
        style,
        batch.input_lengths,
        duration_encoding.size(-1),
    )
    decode_alignment = monotonic
    decode_lengths = [int(value.item() // 2) for value in batch.mel_lengths]
    if stage.validation.duration_source is ValidationDurationSource.PREDICTED:
        decode_alignment, decode_lengths = predicted_alignment(
            duration_predictions,
            batch.input_lengths,
        )
    aligned_text = text_encoding @ decode_alignment
    aligned_duration = duration_encoding @ decode_alignment
    half_lengths = torch.tensor(decode_lengths, device=device)
    predicted_f0, predicted_energy = modules.prosody_predictor(
        aligned_duration,
        style,
        half_lengths,
        decode_alignment.size(-1),
    )
    with torch.autocast(device_type=device.type, enabled=False):
        decoder_voice = modules.voice_encoder.forward_masked(
            batch.reference_mels.unsqueeze(1).float(),
            batch.reference_mel_lengths,
        )
    language_ids = torch.full(
        (batch.texts.size(0),),
        runtime.models.parameters.language_id,
        dtype=torch.long,
        device=device,
    )
    language = modules.language_embedding(language_ids)
    source_lengths = [int(value.item()) for value in batch.mel_lengths]
    full_lengths = [value * 2 for value in decode_lengths]
    resized_f0 = resize_prosody(target_f0, source_lengths, full_lengths)
    resized_energy = resize_prosody(target_energy, source_lengths, full_lengths)
    decoder_f0 = predicted_f0 if stage.validation.f0_source is ProsodySource.PREDICTED else resized_f0
    decoder_energy = predicted_energy if stage.validation.norm_source is ProsodySource.PREDICTED else resized_energy
    positions = torch.arange(max(full_lengths), device=device)
    frame_mask = positions[None, :] < torch.tensor(full_lengths, device=device)[:, None]
    predicted_f0 = predicted_f0 * frame_mask
    predicted_energy = predicted_energy * frame_mask
    reconstructed = modules.decoder(
        aligned_text,
        decoder_f0,
        decoder_energy,
        decoder_voice,
        language,
        frame_mask,
    )
    return (
        reconstructed,
        duration_predictions,
        predicted_f0,
        predicted_energy,
        resized_f0,
        resized_energy,
        decode_alignment,
        decode_lengths,
    )
