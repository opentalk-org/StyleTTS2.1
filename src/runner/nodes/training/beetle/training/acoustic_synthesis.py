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
    f0_prediction_ratio: float
    encoder_kl: Tensor
    f0: Tensor
    n: Tensor
    reconstruction: Tensor
    adversarial: Tensor
    feature_period: Tensor
    feature_resolution: Tensor
    feature_matching: Tensor
    vocoder_total: Tensor
    total: Tensor


@dataclass(frozen=True)
class AcousticTrainingView:
    target: AcousticFeatures
    segment: AlignedSegments
    synthesis: AcousticSynthesis
    predicted_f0_ratio: float


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
    return AlignedSegments.reference_chunks(
        available,
        frame_count,
        acoustic.output_hop,
        generator,
    )


def build_acoustic_training_view(
    acoustic: AcousticModels,
    runtime_seed: int,
    loop: LoopState,
    device: torch.device,
    segment_samples: int,
    waveform: Tensor,
    frame_mask: Tensor,
    predicted_f0_ratio: float,
) -> AcousticTrainingView:
    segment = training_segment(
        frame_mask,
        segment_samples,
        acoustic,
        training_generator(runtime_seed, loop, device, "acoustic", "segment"),
    )
    real = segment.samples(waveform)
    mel = acoustic.reconstruction_loss.transforms[0](real[:, 0])
    jdc_mel = acoustic.jdc_transform(real[:, 0])
    segment_frame_mask = torch.ones(
        mel.shape[0],
        1,
        mel.shape[-1],
        dtype=torch.bool,
        device=mel.device,
    )
    target = acoustic.acoustic_targets(mel, jdc_mel, segment_frame_mask)
    synthesis = synthesize_training_posterior(
        acoustic,
        mel,
        segment_frame_mask,
        target,
        predicted_f0_ratio,
        training_generator(runtime_seed, loop, device, "acoustic", "latent"),
        training_generator(runtime_seed, loop, device, "acoustic", "f0-smoothing"),
        training_generator(runtime_seed, loop, device, "acoustic", "source"),
    )
    return AcousticTrainingView(
        target,
        segment,
        synthesis,
        predicted_f0_ratio,
    )


def acoustic_backward(
    acoustic: AcousticModels,
    runtime: DistributedRuntime,
    optimizer: ScheduledOptimizer,
    accumulation_steps: int,
    waveform: Tensor,
    view: AcousticTrainingView,
    weights: AcousticLossWeights,
    latent_generator: torch.Generator,
    completed_step: int,
) -> AcousticBackwardMetrics:
    real = view.segment.samples(waveform)
    segment_frame_mask = view.synthesis.decoded.mask
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
            view.target.f0,
            segment_frame_mask,
            acoustic.feature_linear.config.f0_scale_hz,
        )
        n = masked_n_smooth_l1(
            posterior.acoustic.n,
            view.target.n,
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
        vocoder_total = (
            reconstruction * weights.reconstruction
            + adversarial_view.adversarial * weights.generator_adversarial
            + adversarial_view.feature_matching * weights.feature_matching
        )
        total = (
            vocoder_total
            + encoder_kl * weights.encoder_kl
            + f0 * weights.f0
            + n * weights.n
        )
    optimizer.backward(total / accumulation_steps)
    return AcousticBackwardMetrics(
        view.predicted_f0_ratio,
        encoder_kl.detach(),
        f0.detach(),
        n.detach(),
        reconstruction.detach(),
        adversarial_view.adversarial.detach(),
        adversarial_view.feature_period.detach(),
        adversarial_view.feature_resolution.detach(),
        adversarial_view.feature_matching.detach(),
        vocoder_total.detach(),
        total.detach(),
    )


def synthesize_training_posterior(
    acoustic_models: AcousticModels,
    mel: Tensor,
    frame_mask: Tensor,
    segment: AlignedSegments,
    latent_generator: torch.Generator,
) -> AcousticSynthesis:
    posterior = acoustic_models.audio_encoder(
        mel,
        frame_mask,
        latent_generator,
    )
    acoustic = acoustic_models.feature_linear(
        posterior.latent,
        posterior.mask,
        frame_mask,
    )
    decoded = acoustic_models.decoder(
        posterior.latent,
        posterior.mask,
        frame_mask,
    )
    waveform = acoustic_models.generator(
        decoded.features,
        decoded.mask,
    )
    sample_mask = frame_mask.repeat_interleave(
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
