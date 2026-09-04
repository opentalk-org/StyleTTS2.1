import logging
import math
from pathlib import Path

import torch
from givemedata_client import GiveMeDataClient

from .config import TrainingConfig
from .tracking import GiveMeDataTracker, MetricValue, TrackerRun
from .val_sample_export import ValidationSampleArtifacts


class TrainingReporter:
    def __init__(self, run: TrackerRun, total_steps: int) -> None:
        self.run = run
        self.total_steps = total_steps

    def log_train(
        self,
        step: int,
        metrics: dict[str, torch.Tensor | float | list[float]],
    ) -> None:
        tracked_metrics: dict[str, MetricValue] = {}
        for name, value in metrics.items():
            if isinstance(value, list):
                tracked_metrics[name] = value
                continue
            scalar = _scalar(value)
            if math.isfinite(scalar):
                metric_name = (
                    name
                    if name.startswith((
                        "overhead/",
                        "performance/",
                        "param_nonfinite/",
                        "grad_nonfinite/",
                    ))
                    else f"train/{name}"
                )
                tracked_metrics[metric_name] = scalar
        tracked_metrics["job_progress"] = max(
            1.0,
            min(99.0, 1.0 + 98.0 * step / self.total_steps),
        )
        self.run.track_metrics(tracked_metrics, step=step)

    def log_validation(
        self,
        step: int,
        metrics: dict[str, torch.Tensor | float],
        samples: list[ValidationSampleArtifacts],
        log_dir: str | Path,
    ) -> None:
        tracked_metrics = {}
        for name, value in metrics.items():
            scalar = _scalar(value)
            if math.isfinite(scalar):
                tracked_metrics[f"val/{name}"] = scalar
        tracked_metrics["val/sample_rows"] = float(len(samples))
        self.run.track_metrics(tracked_metrics, step=step)
        base = Path(log_dir)
        for sample in samples:
            artifact_path = (
                f"validation/step_{step:09d}/{sample.mode}/sample_{sample.index}"
            )
            for relative_path in sample.paths:
                self.run.log_artifact(
                    base / relative_path,
                    artifact_path=artifact_path,
                    step=step,
                )


def start_run(config: TrainingConfig, data_client: GiveMeDataClient) -> TrackerRun:
    logging.getLogger(__name__).info(
        "streaming metrics through givemedata training=%s",
        data_client.run_id,
    )
    return GiveMeDataTracker(data_client.metrics())


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)
