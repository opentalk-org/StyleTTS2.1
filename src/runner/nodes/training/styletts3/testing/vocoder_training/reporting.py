from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import trackio
from matplotlib.figure import Figure

from runner.nodes.training.common.wandb_run import TrackerRun


class EpochReporter:
    def __init__(self, output_dir: Path, run: TrackerRun, sample_rate: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.sample_rate = sample_rate

    def track_train(self, metrics: Mapping[str, float], global_step: int, epoch: int) -> None:
        self._track_metrics("train", metrics, global_step, epoch)

    def track_validation(
        self,
        metrics: Mapping[str, float],
        global_step: int,
        epoch: int,
    ) -> None:
        self._track_metrics("validation", metrics, global_step, epoch)

    def save_validation_item(
        self,
        *,
        epoch: int,
        index: int,
        global_step: int,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        ground_truth_mel: torch.Tensor,
        prediction_mel: torch.Tensor,
    ) -> None:
        item_dir = self.output_dir / f"epoch_{epoch:05d}" / f"val_{index:02d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_array = _audio_array(ground_truth)
        prediction_array = _audio_array(prediction)
        gt_path = item_dir / "gt.wav"
        pred_path = item_dir / "pred.wav"
        image_path = item_dir / "mel.png"
        sf.write(gt_path, ground_truth_array, self.sample_rate, subtype="PCM_16")
        sf.write(pred_path, prediction_array, self.sample_rate, subtype="PCM_16")
        _save_mel_image(image_path, ground_truth_mel, prediction_mel)

        self.run.track(
            trackio.Audio(ground_truth_array, sample_rate=self.sample_rate, caption=f"epoch {epoch} validation {index} ground truth"),
            name=f"validation/audio/{index:02d}/gt",
            step=global_step,
            epoch=epoch,
        )
        self.run.track(
            trackio.Audio(prediction_array, sample_rate=self.sample_rate, caption=f"epoch {epoch} validation {index} prediction"),
            name=f"validation/audio/{index:02d}/pred",
            step=global_step,
            epoch=epoch,
        )
        self.run.track(
            trackio.Image(str(image_path), caption=f"epoch {epoch} validation {index} log-mels"),
            name=f"validation/mel/{index:02d}",
            step=global_step,
            epoch=epoch,
        )

    def close(self) -> None:
        self.run.close()

    def _track_metrics(
        self,
        prefix: str,
        metrics: Mapping[str, float],
        global_step: int,
        epoch: int,
    ) -> None:
        for name, value in metrics.items():
            assert math.isfinite(value), f"non-finite {prefix}/{name}: {value}"
            self.run.track(
                value,
                name=f"{prefix}/{name}",
                step=global_step,
                epoch=epoch,
            )


def _audio_array(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0).numpy()


def _save_mel_image(
    path: Path,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    gt = ground_truth.detach().float().cpu().numpy()
    pred = prediction.detach().float().cpu().numpy()
    lower = float(min(gt.min(), pred.min()))
    upper = float(max(gt.max(), pred.max()))
    figure = Figure(figsize=(12, 6), constrained_layout=True)
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].imshow(gt, origin="lower", aspect="auto", vmin=lower, vmax=upper)
    axes[0].set_title("Ground truth log-mel")
    axes[1].imshow(pred, origin="lower", aspect="auto", vmin=lower, vmax=upper)
    axes[1].set_title("Predicted log-mel")
    axes[1].set_xlabel("Frame")
    for axis in axes:
        axis.set_ylabel("Mel bin")
    figure.savefig(path, dpi=120)
