from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from matplotlib.figure import Figure
from torch import Tensor

from runner.nodes.training.common.mlflow_run import TrackerRun

from .config import SignalConfig


class Reporter:
    def __init__(
        self,
        output_dir: Path,
        run: TrackerRun,
        signal: SignalConfig,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.signal = signal

    def metrics(
        self,
        split: str,
        values: Mapping[str, float],
        step: int,
        epoch: int,
    ) -> None:
        metrics = {}
        for name, value in values.items():
            assert math.isfinite(value), f"non-finite {split}/{name}: {value}"
            metrics[f"{split}/{name}"] = value
        self.run.track_metrics(metrics, step=step, epoch=epoch)

    def validation_sample(
        self,
        step: int,
        index: int,
        target: Tensor,
        prediction: Tensor,
        target_mel: Tensor,
        prediction_mel: Tensor,
    ) -> None:
        directory = self.output_dir / f"step_{step:09d}" / f"sample_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        target_path = directory / "gt.wav"
        prediction_path = directory / "pred.wav"
        mel_path = directory / "mel_spectrogram.png"
        stft_path = directory / "stft_spectrogram.png"
        sf.write(
            target_path,
            audio_array(target),
            self.signal.sample_rate,
            subtype="PCM_16",
        )
        sf.write(
            prediction_path,
            audio_array(prediction),
            self.signal.sample_rate,
            subtype="PCM_16",
        )
        save_comparison(
            mel_path,
            target_mel,
            prediction_mel,
            "Log-mel spectrogram",
        )
        save_comparison(
            stft_path,
            log_stft(target, self.signal),
            log_stft(prediction, self.signal),
            "Log-magnitude STFT spectrogram",
        )
        artifact_path = f"validation/step_{step:09d}/sample_{index:02d}"
        for path in (target_path, prediction_path, mel_path, stft_path):
            self.run.log_artifact(path, artifact_path)

    def close(self) -> None:
        self.run.close()


def audio_array(waveform: Tensor) -> np.ndarray:
    return (
        waveform.detach()
        .float()
        .cpu()
        .reshape(-1)
        .clamp(-1.0, 1.0)
        .numpy()
    )


def log_stft(waveform: Tensor, signal: SignalConfig) -> Tensor:
    window = torch.hann_window(
        signal.win_length,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spectrum = torch.stft(
        waveform.float(),
        n_fft=signal.n_fft,
        hop_length=signal.hop_length,
        win_length=signal.win_length,
        window=window,
        return_complex=True,
    )
    return torch.log(torch.clamp(spectrum.abs(), min=1e-5))


def save_comparison(
    path: Path,
    target: Tensor,
    prediction: Tensor,
    title: str,
) -> None:
    target_array = target.detach().float().cpu().numpy()
    prediction_array = prediction.detach().float().cpu().numpy()
    lower = float(min(target_array.min(), prediction_array.min()))
    upper = float(max(target_array.max(), prediction_array.max()))
    figure = Figure(figsize=(12, 6), constrained_layout=True)
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].imshow(
        target_array,
        origin="lower",
        aspect="auto",
        vmin=lower,
        vmax=upper,
    )
    axes[0].set_title(f"Ground truth {title}")
    axes[1].imshow(
        prediction_array,
        origin="lower",
        aspect="auto",
        vmin=lower,
        vmax=upper,
    )
    axes[1].set_title(f"Prediction {title}")
    axes[1].set_xlabel("Frame")
    for axis in axes:
        axis.set_ylabel("Bin")
    figure.savefig(path, dpi=120)
