from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
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
        self.artifact_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="validation-artifacts",
        )
        self.sample_futures: list[Future[None]] = []
        self.step_futures: list[Future[None]] = []

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
        self.sample_futures.append(
            self.artifact_executor.submit(
                self._write_validation_sample,
                step,
                index,
                target.detach().float().cpu(),
                prediction.detach().float().cpu(),
                target_mel.detach().float().cpu(),
                prediction_mel.detach().float().cpu(),
            )
        )

    def _write_validation_sample(
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

    def validation_step_complete(self, step: int) -> None:
        sample_futures = tuple(self.sample_futures)
        self.sample_futures.clear()
        self.step_futures.append(
            self.artifact_executor.submit(
                self._upload_validation_step,
                step,
                sample_futures,
            )
        )

    def _upload_validation_step(
        self,
        step: int,
        sample_futures: tuple[Future[None], ...],
    ) -> None:
        for future in sample_futures:
            future.result()
        directory = self.output_dir / f"step_{step:09d}"
        self.run.log_artifacts(
            directory,
            f"validation/step_{step:09d}",
        )

    def close(self) -> None:
        try:
            for future in self.sample_futures:
                future.result()
            for future in self.step_futures:
                future.result()
        finally:
            self.artifact_executor.shutdown(wait=True)
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
