from __future__ import annotations

import itertools
import logging
from pathlib import Path
import time

import soundfile as sf
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoints import save_checkpoint
from .config import RunConfig, SignalConfig, TrainingConfig
from .data import AudioEntry
from .discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionSpectralDiscriminator,
)
from .losses import (
    discriminator_loss,
    discriminator_tprls_loss,
    feature_loss,
    generator_tprls_loss,
    generator_loss,
)
from .reporting import Reporter
from .spectral import LogMelSpectrogram

logger = logging.getLogger(__name__)

def discriminator_step(
    real: Tensor,
    fake: Tensor,
    mpd: MultiPeriodDiscriminator,
    mrsd: MultiResolutionSpectralDiscriminator,
    optimizer: torch.optim.Optimizer,
) -> tuple[Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    mpd_real, mpd_fake, _, _ = mpd(real, fake.detach())
    mrsd_real, mrsd_fake, _, _ = mrsd(real, fake.detach())
    period = discriminator_loss(mpd_real, mpd_fake)
    period = period + discriminator_tprls_loss(mpd_real, mpd_fake)
    resolution = discriminator_loss(mrsd_real, mrsd_fake)
    resolution = resolution + discriminator_tprls_loss(
        mrsd_real,
        mrsd_fake,
    )
    (period + resolution).backward()
    optimizer.step()
    return period.detach(), resolution.detach()


def generator_step(
    real: Tensor,
    fake: Tensor,
    mpd: MultiPeriodDiscriminator,
    mrsd: MultiResolutionSpectralDiscriminator,
    mel: LogMelSpectrogram,
    target_mel: Tensor,
    mel_weight: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    set_requires_grad(mpd, False)
    set_requires_grad(mrsd, False)
    try:
        mpd_real, mpd_fake, mpd_real_maps, mpd_fake_maps = mpd(real, fake)
        mrsd_real, mrsd_fake, mrsd_real_maps, mrsd_fake_maps = mrsd(real, fake)
        period_features = feature_loss(mpd_real_maps, mpd_fake_maps)
        resolution_features = feature_loss(mrsd_real_maps, mrsd_fake_maps)
        period_adversarial = generator_loss(mpd_fake)
        period_adversarial = period_adversarial + generator_tprls_loss(
            mpd_real,
            mpd_fake,
        )
        resolution_adversarial = generator_loss(mrsd_fake)
        resolution_adversarial = resolution_adversarial + generator_tprls_loss(
            mrsd_real,
            mrsd_fake,
        )
        mel_error = F.l1_loss(mel(fake.squeeze(1)), target_mel)
        total = (
            period_features
            + resolution_features
            + period_adversarial
            + resolution_adversarial
            + mel_error * mel_weight
        )
        total.backward()
        optimizer.step()
    finally:
        set_requires_grad(mpd, True)
        set_requires_grad(mrsd, True)
    return {
        "generator": float(total.detach()),
        "mel_spec_error": float(mel_error.detach()),
        "feature_period": float(period_features.detach()),
        "feature_resolution": float(resolution_features.detach()),
        "adversarial_period": float(period_adversarial.detach()),
        "adversarial_resolution": float(resolution_adversarial.detach()),
    }


def train_batch(
    waveform: Tensor,
    generator: nn.Module,
    mpd: MultiPeriodDiscriminator,
    mrsd: MultiResolutionSpectralDiscriminator,
    mel: LogMelSpectrogram,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    target = waveform.to(device, non_blocking=True)
    real = target.unsqueeze(1)
    target_mel = mel(target)
    generated_spec, generated_phase = generator(target_mel)
    fake = generator.stft.inverse(generated_spec, generated_phase)
    period, resolution = discriminator_step(
        real,
        fake,
        mpd,
        mrsd,
        discriminator_optimizer,
    )
    metrics = generator_step(
        real,
        fake,
        mpd,
        mrsd,
        mel,
        target_mel,
        config.mel_weight,
        generator_optimizer,
    )
    metrics["discriminator_period"] = float(period)
    metrics["discriminator_resolution"] = float(resolution)
    return metrics


def parameter_l2_norm(module: nn.Module) -> float:
    parameters = tuple(module.parameters())
    squared_norm = torch.zeros(
        (),
        device=parameters[0].device,
        dtype=torch.float32,
    )
    for parameter in parameters:
        squared_norm += parameter.detach().float().square().sum()
    return float(torch.sqrt(squared_norm))


def weight_norm_metrics(
    generator: nn.Module,
    mpd: MultiPeriodDiscriminator,
    mrsd: MultiResolutionSpectralDiscriminator,
) -> dict[str, float]:
    metrics = {
        "weight_norm/generator": parameter_l2_norm(generator),
        "weight_norm/mpd": parameter_l2_norm(mpd),
        "weight_norm/mrsd": parameter_l2_norm(mrsd),
    }
    metrics.update(
        {
            f"weight_norm/mpd_period_{discriminator.period}": parameter_l2_norm(
                discriminator
            )
            for discriminator in mpd.discriminators
        }
    )
    metrics.update(
        {
            f"weight_norm/mrsd_fft_{discriminator.n_fft}": parameter_l2_norm(
                discriminator
            )
            for discriminator in mrsd.discriminators
        }
    )
    return metrics


def train(
    generator: nn.Module,
    mpd: MultiPeriodDiscriminator,
    mrsd: MultiResolutionSpectralDiscriminator,
    loader: DataLoader[Tensor],
    validation: list[AudioEntry],
    signal: SignalConfig,
    config: TrainingConfig,
    run_config: RunConfig,
    output_dir: Path,
    reporter: Reporter,
    device: torch.device,
) -> None:
    if len(loader) == 0:
        raise ValueError("training dataset has fewer segments than one batch")
    generator.to(device)
    mpd.to(device)
    mrsd.to(device)
    mel = LogMelSpectrogram(signal).to(device)
    generator_optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in generator.parameters()
            if parameter.requires_grad
        ),
        config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    discriminator_optimizer = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), mrsd.parameters()),
        config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        generator_optimizer,
        config.learning_rate_decay,
    )
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        discriminator_optimizer,
        config.learning_rate_decay,
    )
    step = 0
    for epoch in range(1, run_config.epochs + 1):
        generator.train()
        mpd.train()
        mrsd.train()
        for waveform in loader:
            metrics = train_batch(
                waveform,
                generator,
                mpd,
                mrsd,
                mel,
                generator_optimizer,
                discriminator_optimizer,
                config,
                device,
            )
            step += 1
            if step % config.weight_norm_interval == 0:
                metrics.update(weight_norm_metrics(generator, mpd, mrsd))
            reporter.metrics("train", metrics, step, epoch)
            logger.info(
                "epoch=%d step=%d generator=%.4f discriminator=%.4f",
                epoch,
                step,
                metrics["generator"],
                metrics["discriminator_period"]
                + metrics["discriminator_resolution"],
            )
            if step % run_config.checkpoint_interval == 0:
                save_checkpoint(
                    output_dir,
                    step,
                    epoch,
                    generator,
                    mpd,
                    mrsd,
                    generator_optimizer,
                    discriminator_optimizer,
                )
            if step % run_config.validation_interval == 0:
                validate(
                    generator,
                    validation,
                    mel,
                    reporter,
                    signal,
                    device,
                    step,
                    epoch,
                )
            if run_config.max_steps is not None and step >= run_config.max_steps:
                save_checkpoint(
                    output_dir,
                    step,
                    epoch,
                    generator,
                    mpd,
                    mrsd,
                    generator_optimizer,
                    discriminator_optimizer,
                )
                return
        generator_scheduler.step()
        discriminator_scheduler.step()
    save_checkpoint(
        output_dir,
        step,
        run_config.epochs,
        generator,
        mpd,
        mrsd,
        generator_optimizer,
        discriminator_optimizer,
    )


def validate(
    generator: nn.Module,
    entries: list[AudioEntry],
    mel: LogMelSpectrogram,
    reporter: Reporter,
    signal: SignalConfig,
    device: torch.device,
    step: int,
    epoch: int,
) -> None:
    generator.eval()
    started = time.perf_counter()
    errors = []
    with torch.no_grad():
        for index, entry in enumerate(entries):
            frames, sample_rate = sf.read(entry.path, dtype="float32")
            assert sample_rate == signal.sample_rate
            waveform = torch.from_numpy(frames).float().unsqueeze(0)
            original_length = waveform.shape[-1]
            waveform = F.pad(
                waveform,
                (0, (-original_length) % signal.hop_length),
            ).to(device)
            target_mel = mel(waveform)
            generated_spec, generated_phase = generator(target_mel)
            prediction = generator.stft.inverse(
                generated_spec,
                generated_phase,
            )
            prediction_mel = mel(prediction.squeeze(1))
            errors.append(float(F.l1_loss(prediction_mel, target_mel)))
            reporter.validation_sample(
                step,
                index,
                waveform[0, :original_length],
                prediction[0, 0, :original_length],
                target_mel[0],
                prediction_mel[0],
            )
    reporter.validation_step_complete(step)
    reporter.metrics(
        "validation",
        {"mel_spec_error": sum(errors) / len(errors)},
        step,
        epoch,
    )
    logger.info(
        "validation step=%d samples=%d seconds=%.2f",
        step,
        len(entries),
        time.perf_counter() - started,
    )
    generator.train()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
