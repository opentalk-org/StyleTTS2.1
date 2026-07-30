import torch

from ...stages import (
    ProsodySource,
    TrainingStageSpec,
    ValidationDurationSource,
)
from ..data import TrainingBatch
from ..setup import TrainingRuntime
from ..utils import log_norm
from .validation_batch import (
    ValidationBatch,
    predicted_alignment,
    resize_prosody,
)


def synthesize_validation(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    stage: TrainingStageSpec,
    validation_batch: ValidationBatch,
    text_encoding: torch.Tensor,
    duration_encoding: torch.Tensor,
    bert: torch.Tensor,
    text_mask: torch.Tensor,
    monotonic: torch.Tensor,
    duration_style: torch.Tensor,
    acoustic_style: torch.Tensor,
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
    if stage.validation.diffusion:
        style_reference = None
        if runtime.models.parameters.multispeaker:
            style_reference = torch.cat(
                (
                    modules.style_encoder(batch.reference_mels.unsqueeze(1)),
                    modules.predictor_encoder(
                        batch.reference_mels.unsqueeze(1)
                    ),
                ),
                dim=1,
            )
        style_target = torch.cat((acoustic_style, duration_style), dim=1)
        sampler_arguments = {
            "noise": torch.randn_like(style_target).unsqueeze(1),
            "embedding": bert,
            "embedding_scale": 1,
            "num_steps": 5,
        }
        if style_reference is not None:
            sampler_arguments["features"] = style_reference
        diffusion_style = runtime.diffusion_sampler(
            **sampler_arguments
        ).squeeze(1)
        acoustic_style = diffusion_style[:, :128]
        duration_style = diffusion_style[:, 128:]

    duration_predictions, _, predicted_f0, predicted_norm = modules.predictor(
        duration_encoding,
        duration_style,
        batch.input_lengths,
        monotonic,
        text_mask,
        validation_batch.starts,
        validation_batch.frames,
        duration_style,
    )
    decode_alignment = monotonic
    decode_lengths = validation_batch.lengths
    decode_aligned_text = validation_batch.aligned_text
    if (
        stage.validation.duration_source
        is ValidationDurationSource.PREDICTED
    ):
        decode_alignment, decode_lengths = predicted_alignment(
            duration_predictions,
            batch.input_lengths,
        )
        decode_aligned_text = text_encoding @ decode_alignment
        duration_features = modules.predictor.text_encoder(
            duration_encoding,
            duration_style,
            batch.input_lengths,
            text_mask,
        )
        duration_prosody = (
            duration_features.transpose(-1, -2) @ decode_alignment
        )
        predicted_f0, predicted_norm = modules.predictor.F0Ntrain(
            duration_prosody,
            duration_style,
        )

    target_f0, _, _ = modules.pitch_extractor(
        validation_batch.mel.unsqueeze(1)
    )
    target_f0 = resize_prosody(
        target_f0.squeeze(-1),
        validation_batch.lengths,
        decode_lengths,
    )
    target_norm = resize_prosody(
        log_norm(validation_batch.mel.unsqueeze(1)).squeeze(1),
        validation_batch.lengths,
        decode_lengths,
    )
    decoder_f0 = (
        predicted_f0
        if stage.validation.f0_source is ProsodySource.PREDICTED
        else target_f0
    )
    decoder_norm = (
        predicted_norm
        if stage.validation.norm_source is ProsodySource.PREDICTED
        else target_norm
    )
    reconstructed = modules.decoder(
        decode_aligned_text,
        decoder_f0,
        decoder_norm,
        acoustic_style,
    )
    return (
        reconstructed,
        duration_predictions,
        predicted_f0,
        predicted_norm,
        target_f0,
        target_norm,
        decode_alignment,
        decode_lengths,
    )
