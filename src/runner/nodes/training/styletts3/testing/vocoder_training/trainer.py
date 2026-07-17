from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from runner.nodes.training.styletts3.testing.discriminator_backend import VocoderDiscriminator
from runner.nodes.training.styletts3.testing.vocoder_training.audio_data import (
    AudioEntry,
    read_full_audio,
)
from runner.nodes.training.styletts3.testing.vocoder_training.mel import (
    LogMelSpectrogram,
    MultiResolutionMelLoss,
    conditioning_mel,
    pad_to_hop,
)
from runner.nodes.training.styletts3.testing.vocoder_training.optimizers import (
    build_adam_optimizers,
)
from runner.nodes.training.styletts3.testing.vocoder_training.progress import (
    TrainingProgressEstimator,
    overhead_percent,
)
from runner.nodes.training.styletts3.testing.vocoder_training.profiles import SignalGeometry
from runner.nodes.training.styletts3.testing.vocoder_training.reporting import EpochReporter
from runner.nodes.training.styletts3.testing.vocoder_training.system_metrics import (
    SystemMetricsSampler,
)
from runner.nodes.training.styletts3.testing.vocoder_training.training_metrics import (
    gradient_l2_norm,
    mean_logit,
    mean_metrics,
)
from runner.nodes.training.styletts3.testing.vocoder_training.training_config import TrainingConfig
from runner.nodes.training.styletts3.testing.vocoder_training.validation_runtime import (
    validation_cudnn_benchmark_disabled,
)

logger = logging.getLogger(__name__)
MEL_WEIGHT = 45.0
FEATURE_MATCHING_WEIGHT = 2.0
PROGRESS_WINDOW_STEPS = 100
PROGRESS_WARMUP_INTERVALS = 32
SYSTEM_METRICS_INTERVAL_SECONDS = 10.0

def train_batch(
    generator: nn.Module,
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
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=discriminator.use_autocast,
    ):
        real_evaluation, fake_evaluation = discriminator.evaluate_pair(real_3d, fake_3d.detach())
        d_loss = discriminator.discriminator_loss(real_evaluation, fake_evaluation)
    d_loss.backward()
    discriminator_gradient_norm = gradient_l2_norm(discriminator)
    discriminator_optimizer.step()

    _set_requires_grad(discriminator, False)
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=discriminator.use_autocast,
        ):
            real_evaluation, fake_evaluation = discriminator.evaluate_pair(real_3d, fake_3d)
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
        generator_gradient_norm = gradient_l2_norm(generator)
        generator_optimizer.step()
    finally:
        _set_requires_grad(discriminator, True)

    return {
        "generator": float(g_loss.detach()),
        "discriminator": float(d_loss.detach()),
        "mel": float(reconstruction.detach()),
        "adversarial": float(adversarial.detach()),
        "feature_matching": float(feature_matching.detach()),
        "generator_gradient_norm": generator_gradient_norm,
        "discriminator_gradient_norm": discriminator_gradient_norm,
        "real_logit_mean": mean_logit(real_evaluation.logits),
        "fake_logit_mean": mean_logit(fake_evaluation.logits),
    }


def validate_epoch(
    generator: nn.Module,
    discriminator: VocoderDiscriminator,
    conditioner: LogMelSpectrogram,
    mel_loss: MultiResolutionMelLoss,
    entries: list[AudioEntry],
    device: torch.device,
    reporter: EpochReporter,
    epoch: int,
    global_step: int,
    signal: SignalGeometry,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for index, entry in enumerate(entries):
            waveform = read_full_audio(entry, signal).unsqueeze(0)
            padded, original_length = pad_to_hop(waveform, signal.synthesis_hop)
            real = padded.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                condition = conditioner(real)
                fake_3d = generator(condition)
                real_evaluation, fake_evaluation = discriminator.evaluate_pair(real.unsqueeze(1), fake_3d)
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
    return mean_metrics(rows)


def train_vocoder(
    generator: nn.Module,
    discriminator: VocoderDiscriminator,
    train_loader: DataLoader[torch.Tensor],
    validation_entries: list[AudioEntry],
    config: TrainingConfig,
    device: torch.device,
    reporter: EpochReporter,
    signal: SignalGeometry,
) -> None:
    loader_steps = len(train_loader)
    if loader_steps == 0:
        raise ValueError("training loader has no complete batch")
    steps_per_epoch = config.effective_steps_per_epoch(loader_steps)
    total_steps = config.total_steps(loader_steps)
    generator.to(device)
    discriminator.to(device)
    conditioner = conditioning_mel(signal).to(device)
    mel_loss = MultiResolutionMelLoss(signal).to(device)
    generator_optimizer, discriminator_optimizer = build_adam_optimizers(
        generator,
        discriminator,
        config.generator_learning_rate,
        config.discriminator_learning_rate,
        config.betas,
    )
    global_step = 0
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
    for epoch in range(1, config.training_epochs(loader_steps) + 1):
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
            metrics["generator_learning_rate"] = config.generator_learning_rate
            metrics["discriminator_learning_rate"] = config.discriminator_learning_rate
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
            if batch_index >= steps_per_epoch or global_step >= total_steps:
                break
        epoch_train = {f"epoch_{name}": value for name, value in mean_metrics(train_rows).items()}
        reporter.track_train(epoch_train, global_step, epoch)
        if epoch % config.validation_interval_epochs == 0:
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
                    signal,
                )
            reporter.track_validation(validation, global_step, epoch)
    torch.save(generator.state_dict(), reporter.output_dir / "generator_final.pth")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
