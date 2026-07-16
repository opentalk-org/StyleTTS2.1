from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from runner.nodes.training.styletts3.testing.discriminator_backend import VocoderDiscriminator
from runner.nodes.training.styletts3.testing.istftnet2_mb import ISTFTNet2MB
from runner.nodes.training.styletts3.testing.vocoder_training.audio_data import AudioEntry
from runner.nodes.training.styletts3.testing.vocoder_training.geometry import SAMPLE_RATE
from runner.nodes.training.styletts3.testing.vocoder_training.mel import (
    LogMelSpectrogram,
    MultiResolutionMelLoss,
    conditioning_mel,
    pad_to_hop,
)
from runner.nodes.training.styletts3.testing.vocoder_training.progress import (
    TrainingProgressEstimator,
    overhead_percent,
)
from runner.nodes.training.styletts3.testing.vocoder_training.reporting import EpochReporter
from runner.nodes.training.styletts3.testing.vocoder_training.system_metrics import (
    SystemMetricsSampler,
)

logger = logging.getLogger(__name__)
MEL_WEIGHT = 45.0
FEATURE_MATCHING_WEIGHT = 2.0
PROGRESS_WINDOW_STEPS = 100
PROGRESS_WARMUP_INTERVALS = 32
SYSTEM_METRICS_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    betas: tuple[float, float]
    max_steps_per_epoch: int | None
    validation_interval_epochs: int


def is_validation_epoch(epoch: int, interval: int) -> bool:
    """Return whether this epoch owns the expensive full-audio validation pass."""
    return epoch % interval == 0


@contextmanager
def validation_cudnn_benchmark_disabled() -> Iterator[None]:
    """Avoid per-shape autotuning for variable-length validation recordings."""
    enabled = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = enabled


def train_batch(
    generator: ISTFTNet2MB,
    discriminator: VocoderDiscriminator,
    conditioner: LogMelSpectrogram,
    mel_loss: MultiResolutionMelLoss,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    waveform: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    real = waveform.to(device, non_blocking=True)
    real_3d = real.unsqueeze(1)
    generator_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        condition = conditioner(real)
        fake_3d = generator(condition)

    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        real_evaluation = discriminator.evaluate(real_3d)
        fake_evaluation = discriminator.evaluate(fake_3d.detach())
        d_loss = discriminator.discriminator_loss(real_evaluation, fake_evaluation)
    d_loss.backward()
    discriminator_optimizer.step()

    _set_requires_grad(discriminator, False)
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                real_evaluation = discriminator.evaluate(real_3d)
            fake_evaluation = discriminator.evaluate(fake_3d)
            adversarial = discriminator.generator_adv_loss(real_evaluation, fake_evaluation)
            feature_matching = discriminator.feature_matching_loss(
                real_evaluation,
                fake_evaluation,
            )
            reconstruction = mel_loss(fake_3d.squeeze(1), real)
            g_loss = (
                adversarial
                + FEATURE_MATCHING_WEIGHT * feature_matching
                + MEL_WEIGHT * reconstruction
            )
        g_loss.backward()
        generator_optimizer.step()
    finally:
        _set_requires_grad(discriminator, True)

    return {
        "generator": float(g_loss.detach()),
        "discriminator": float(d_loss.detach()),
        "mel": float(reconstruction.detach()),
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
    }


def validate_epoch(
    generator: ISTFTNet2MB,
    discriminator: VocoderDiscriminator,
    conditioner: LogMelSpectrogram,
    mel_loss: MultiResolutionMelLoss,
    entries: list[AudioEntry],
    device: torch.device,
    reporter: EpochReporter,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for index, entry in enumerate(entries):
            waveform = _read_full_audio(entry).unsqueeze(0)
            padded, original_length = pad_to_hop(waveform)
            real = padded.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                condition = conditioner(real)
                fake_3d = generator(condition)
                real_evaluation = discriminator.evaluate(real.unsqueeze(1))
                fake_evaluation = discriminator.evaluate(fake_3d)
                reconstruction = mel_loss(fake_3d.squeeze(1), real)
                adversarial = discriminator.generator_adv_loss(
                    real_evaluation,
                    fake_evaluation,
                )
                feature_matching = discriminator.feature_matching_loss(
                    real_evaluation,
                    fake_evaluation,
                )
                d_loss = discriminator.discriminator_loss(real_evaluation, fake_evaluation)
                waveform_l1 = torch.nn.functional.l1_loss(fake_3d.squeeze(1), real)
                g_loss = (
                    adversarial
                    + FEATURE_MATCHING_WEIGHT * feature_matching
                    + MEL_WEIGHT * reconstruction
                )
                prediction_mel = conditioner(fake_3d.squeeze(1))
            prediction = fake_3d[0, 0, :original_length]
            ground_truth = real[0, :original_length]
            reporter.save_validation_item(
                epoch=epoch,
                index=index,
                global_step=global_step,
                ground_truth=ground_truth,
                prediction=prediction,
                ground_truth_mel=condition[0],
                prediction_mel=prediction_mel[0],
            )
            rows.append(
                {
                    "generator": float(g_loss),
                    "discriminator": float(d_loss),
                    "mel": float(reconstruction),
                    "waveform_l1": float(waveform_l1),
                    "adversarial": float(adversarial),
                    "feature_matching": float(feature_matching),
                }
            )
    return _mean_metrics(rows)


def train_vocoder(
    generator: ISTFTNet2MB,
    discriminator: VocoderDiscriminator,
    train_loader: DataLoader[torch.Tensor],
    validation_entries: list[AudioEntry],
    config: TrainingConfig,
    device: torch.device,
    reporter: EpochReporter,
) -> None:
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("training loader has no complete batch")
    generator.to(device)
    discriminator.to(device)
    conditioner = conditioning_mel().to(device)
    mel_loss = MultiResolutionMelLoss().to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=config.betas, fused=True
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=config.betas, fused=True
    )
    global_step = 0
    total_steps = steps_per_epoch * config.epochs
    progress = TrainingProgressEstimator(
        total_steps,
        PROGRESS_WINDOW_STEPS,
        PROGRESS_WARMUP_INTERVALS,
    )
    system_metrics = SystemMetricsSampler(
        torch.cuda.current_device(),
        SYSTEM_METRICS_INTERVAL_SECONDS,
        time.perf_counter(),
    )
    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()
        train_rows: list[dict[str, float]] = []
        for batch_index, waveform in enumerate(train_loader, start=1):
            started = time.perf_counter()
            if epoch > 1 and batch_index == 1:
                progress.resume(started)
            metrics = train_batch(
                generator,
                discriminator,
                conditioner,
                mel_loss,
                generator_optimizer,
                discriminator_optimizer,
                waveform,
                device,
            )
            elapsed = time.perf_counter() - started
            global_step += 1
            metrics["examples_per_second"] = waveform.shape[0] / elapsed
            metrics["samples_per_second"] = waveform.numel() / elapsed
            metrics["learning_rate"] = config.learning_rate
            progress_metrics = progress.update(global_step, time.perf_counter())
            metrics.update(progress_metrics)
            if progress_metrics:
                metrics["overhead_percent"] = overhead_percent(
                    metrics["examples_per_second"],
                    waveform.shape[0],
                    progress_metrics["steps_per_second"],
                )
            sampled_system_metrics = system_metrics.sample(time.perf_counter())
            train_rows.append(metrics)
            reporter.track_train(metrics, global_step, epoch)
            if sampled_system_metrics:
                reporter.track_system(sampled_system_metrics, global_step, epoch)
            logger.info(
                "epoch=%d step=%d generator=%.4f discriminator=%.4f",
                epoch,
                global_step,
                metrics["generator"],
                metrics["discriminator"],
            )
            if config.max_steps_per_epoch is not None and batch_index >= config.max_steps_per_epoch:
                break
        epoch_train = {f"epoch_{name}": value for name, value in _mean_metrics(train_rows).items()}
        reporter.track_train(epoch_train, global_step, epoch)
        if is_validation_epoch(epoch, config.validation_interval_epochs):
            with validation_cudnn_benchmark_disabled():
                validation = validate_epoch(
                    generator,
                    discriminator,
                    conditioner,
                    mel_loss,
                    validation_entries,
                    device,
                    reporter,
                    epoch,
                    global_step,
                )
            reporter.track_validation(validation, global_step, epoch)
    torch.save(generator.state_dict(), reporter.output_dir / "generator_final.pth")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _read_full_audio(entry: AudioEntry) -> torch.Tensor:
    frames, sample_rate = sf.read(entry.path, dtype="float32", always_2d=True)
    assert sample_rate == SAMPLE_RATE, f"validation sample rate changed: {entry.path}"
    return torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    assert rows, "metric rows must not be empty"
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
