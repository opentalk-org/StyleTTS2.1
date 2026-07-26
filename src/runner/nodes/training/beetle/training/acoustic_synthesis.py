from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.acoustic import (
    masked_f0_smooth_l1,
    masked_kl_standard_normal,
    masked_n_smooth_l1,
)
from ..losses.adversarial import generator_step_loss
from ..losses.composition import AcousticLossWeights
from ..models.model import AcousticModels, AcousticSynthesis
from ..models.modules.audio import AcousticFeatures, AudioPosterior
from ..models.modules.segments import AlignedSegments
from .distributed import DistributedRuntime
from .optimizer import ScheduledOptimizer
from .state import LoopState


@dataclass(frozen=True)
class AcousticBackwardMetrics:
    prediction_ratio: float
    encoder_kl: Tensor
    f0: Tensor
    n: Tensor
    reconstruction: Tensor
    adversarial: Tensor
    feature_matching: Tensor
    total: Tensor


def batch_inputs(
    batch: BeetleBatch,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    waveform = batch.waveform.to(device, non_blocking=True)
    mel = batch.mel.to(device, non_blocking=True)
    return waveform, mel, batch.frame_mask.to(device, non_blocking=True)


def training_generator(
    runtime_seed: int,
    loop: LoopState,
    device: torch.device,
    view: str,
    purpose: str,
) -> torch.Generator:
    seed = derive_seed(
        runtime_seed,
        loop.cycle,
        loop.batch_index,
        view,
        purpose,
    )
    return torch.Generator(device=device).manual_seed(seed)


def training_segment(
    frame_mask: Tensor,
    segment_samples: int,
    acoustic: AcousticModels,
    generator: torch.Generator,
) -> AlignedSegments:
    frame_count = segment_samples // acoustic.output_hop
    lengths = frame_mask[:, 0].sum(dim=1).clamp_min(frame_count)
    positions = torch.arange(frame_mask.shape[-1], device=frame_mask.device)
    available = positions.view(1, 1, -1) < lengths.view(-1, 1, 1)
    return AlignedSegments.random(
        available,
        frame_count,
        acoustic.latent_downsample_rate,
        acoustic.output_hop,
        generator,
    )


def acoustic_backward(
    acoustic: AcousticModels,
    runtime: DistributedRuntime,
    optimizer: ScheduledOptimizer,
    accumulation_steps: int,
    waveform: Tensor,
    mel: Tensor,
    frame_mask: Tensor,
    target: AcousticFeatures,
    segment: AlignedSegments,
    predicted_ratio: float,
    weights: AcousticLossWeights,
    latent_generator: torch.Generator,
    completed_step: int,
) -> AcousticBackwardMetrics:
    real = segment.samples(waveform)
    segment_frame_mask = segment.frames(frame_mask)
    with runtime.autocast():
        posterior = synthesize_training_posterior(
            acoustic,
            mel,
            frame_mask,
            segment,
            latent_generator,
        )
        encoder_kl = masked_kl_standard_normal(
            posterior.posterior.mean,
            posterior.posterior.log_scale,
            posterior.posterior.mask,
        )
        f0 = masked_f0_smooth_l1(
            posterior.acoustic.f0,
            segment.frames(target.f0),
            segment_frame_mask,
            acoustic.feature_linear.config.f0_scale_hz,
        )
        n = masked_n_smooth_l1(
            posterior.acoustic.n,
            segment.frames(target.n),
            segment_frame_mask,
        )
        reconstruction = acoustic.reconstruction_loss(
            posterior.waveform,
            real,
            posterior.sample_mask,
            completed_step,
        ).total
        adversarial_view = generator_step_loss(
            acoustic.discriminators,
            real,
            posterior.waveform,
        )
        total = (
            encoder_kl * weights.encoder_kl
            + f0 * weights.f0
            + n * weights.n
            + reconstruction * weights.reconstruction
            + adversarial_view.adversarial * weights.generator_adversarial
            + adversarial_view.feature_matching * weights.feature_matching
        )
    optimizer.backward(total / accumulation_steps)
    return AcousticBackwardMetrics(
        predicted_ratio,
        encoder_kl.detach(),
        f0.detach(),
        n.detach(),
        reconstruction.detach(),
        adversarial_view.adversarial.detach(),
        adversarial_view.feature_matching.detach(),
        total.detach(),
    )


def synthesize_training_posterior(
    acoustic_models: AcousticModels,
    mel: Tensor,
    frame_mask: Tensor,
    segment: AlignedSegments,
    latent_generator: torch.Generator,
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
        posterior.latent.detach(),
        posterior.mask,
        segment_frame_mask,
    )
    decoded = acoustic_models.decoder(
        posterior.latent,
        posterior.mask,
        segment_frame_mask,
    )
    waveform = acoustic_models.generator(
        decoded.features,
        decoded.mask,
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
