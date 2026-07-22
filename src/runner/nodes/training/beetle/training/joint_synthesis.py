import torch
from torch import Tensor

from ..models.model import AcousticModels, AcousticSynthesis
from ..models.modules.audio import AcousticFeatures, AudioPosterior
from ..models.modules.decoder import DecoderOutput
from ..models.modules.latent_flow import integrate_latent_flow
from ..models.modules.segments import AlignedSegments
from ..models.conditional import ConditionalModels
from .conditional_features import ConditionalSynthesis, ConditionalSynthesisInput


def synthesize_training_pair(
    acoustic_models: AcousticModels,
    conditional_models: ConditionalModels,
    inputs: ConditionalSynthesisInput,
    mel: Tensor,
    frame_mask: Tensor,
    segment: AlignedSegments,
    target: AcousticFeatures,
    predicted_ratio: float,
    latent_generator: torch.Generator,
    source_generator: torch.Generator,
) -> tuple[AcousticSynthesis, ConditionalSynthesis]:
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
    posterior_acoustic = acoustic_models.feature_linear(
        posterior.latent,
        posterior.mask,
        segment_frame_mask,
    )
    conditional_latent = integrate_latent_flow(
        conditional_models.latent_flow,
        inputs.noise,
        inputs.conditions,
        inputs.latent_mask,
        1,
    )
    conditional_latent_segment = segment.latents(conditional_latent)
    conditional_mask = segment.latents(inputs.latent_mask)
    conditional_acoustic = acoustic_models.feature_linear(
        conditional_latent_segment,
        conditional_mask,
        segment_frame_mask,
    )
    segment_target = AcousticFeatures(
        segment.frames(target.f0),
        segment.frames(target.n),
    )
    posterior_decoder_acoustic = segment_target.blend(
        posterior_acoustic,
        predicted_ratio,
    )
    conditional_decoder_acoustic = segment_target.blend(
        conditional_acoustic,
        predicted_ratio,
    )
    joint_frame_mask = torch.cat((segment_frame_mask, segment_frame_mask), dim=0)
    decoded = acoustic_models.decoder(
        torch.cat(
            (
                posterior.latent,
                conditional_latent_segment,
            ),
            dim=0,
        ),
        torch.cat(
            (
                posterior_decoder_acoustic.f0,
                conditional_decoder_acoustic.f0,
            ),
            dim=0,
        ),
        torch.cat(
            (
                posterior_decoder_acoustic.n,
                conditional_decoder_acoustic.n,
            ),
            dim=0,
        ),
        torch.cat(
            (
                posterior.mask,
                conditional_mask,
            ),
            dim=0,
        ),
        joint_frame_mask,
    )
    waveform = acoustic_models.generator(
        decoded.features,
        decoded.f0,
        decoded.mask,
        source_generator,
    )
    batch_size = mel.shape[0]
    sample_mask = segment_frame_mask.repeat_interleave(
        acoustic_models.output_hop,
        dim=-1,
    )
    posterior_synthesis = AcousticSynthesis(
        posterior,
        posterior_acoustic,
        _slice_decoded(decoded, 0, batch_size),
        waveform[:batch_size],
        sample_mask,
    )
    conditional_synthesis = ConditionalSynthesis(
        conditional_acoustic,
        _slice_decoded(decoded, batch_size, batch_size * 2),
        waveform[batch_size:],
        sample_mask,
    )
    return posterior_synthesis, conditional_synthesis


def _slice_decoded(decoded: DecoderOutput, start: int, end: int) -> DecoderOutput:
    return DecoderOutput(
        decoded.features[start:end],
        decoded.f0[start:end],
        decoded.n[start:end],
        decoded.mask[start:end],
    )


def _slice_posterior(posterior: AudioPosterior, start: int, end: int) -> AudioPosterior:
    return AudioPosterior(
        posterior.mean[:, :, start:end],
        posterior.log_scale[:, :, start:end],
        posterior.latent[:, :, start:end],
        posterior.mask[:, :, start:end],
    )
