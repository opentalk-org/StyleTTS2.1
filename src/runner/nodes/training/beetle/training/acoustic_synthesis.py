import torch
from torch import Tensor

from ..models.model import AcousticModels, AcousticSynthesis
from ..models.modules.audio import AcousticFeatures, AudioPosterior
from ..models.modules.segments import AlignedSegments


def synthesize_training_posterior(
    acoustic_models: AcousticModels,
    mel: Tensor,
    frame_mask: Tensor,
    segment: AlignedSegments,
    target: AcousticFeatures,
    predicted_ratio: float,
    latent_generator: torch.Generator,
    source_generator: torch.Generator,
) -> AcousticSynthesis:
    segment_frame_mask = segment.frames(frame_mask)
    encoder_mel = segment.context_frames(
        mel,
        acoustic_models.encoder_context_frames,
    )
    encoder_mask = segment.context_frames(
        frame_mask,
        acoustic_models.encoder_context_frames,
    )
    posterior_window = acoustic_models.audio_encoder(
        encoder_mel,
        encoder_mask,
        latent_generator,
    )
    posterior_start = (
        acoustic_models.encoder_context_frames
        // 2
        // acoustic_models.latent_downsample_rate
    )
    posterior_count = segment.frame_count // acoustic_models.latent_downsample_rate
    posterior = _slice_posterior(
        posterior_window,
        posterior_start,
        posterior_start + posterior_count,
    )
    acoustic = acoustic_models.feature_linear(
        posterior.latent,
        posterior.mask,
        segment_frame_mask,
    )
    segment_target = AcousticFeatures(
        segment.frames(target.f0),
        segment.frames(target.n),
    )
    decoder_acoustic = segment_target.blend(acoustic, predicted_ratio)
    decoded = acoustic_models.decoder(
        posterior.latent,
        decoder_acoustic.f0,
        decoder_acoustic.n,
        posterior.mask,
        segment_frame_mask,
    )
    waveform = acoustic_models.generator(
        decoded.features,
        decoded.f0,
        decoded.mask,
        source_generator,
    )
    sample_mask = segment_frame_mask.repeat_interleave(
        acoustic_models.output_hop,
        dim=-1,
    )
    return AcousticSynthesis(
        posterior,
        acoustic,
        decoded,
        waveform,
        sample_mask,
    )


def _slice_posterior(
    posterior: AudioPosterior,
    start: int,
    end: int,
) -> AudioPosterior:
    return AudioPosterior(
        posterior.mean[:, :, start:end],
        posterior.log_scale[:, :, start:end],
        posterior.latent[:, :, start:end],
        posterior.mask[:, :, start:end],
    )
