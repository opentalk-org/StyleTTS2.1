import math
from pathlib import Path

import torch

from runner.nodes.training.common.mlflow_run import (
    TrackerRun,
    start_mlflow_run,
)

from ..studio.val_sample_export import ValidationSampleArtifacts
from .config import TrainingConfig


class MlflowLogger:
    def __init__(self, run: TrackerRun, total_steps: int) -> None:
        self.run = run
        self.total_steps = total_steps

    def log_train(
        self,
        step: int,
        metrics: dict[str, torch.Tensor | float],
    ) -> None:
        tracked_metrics = {}
        for name, value in metrics.items():
            scalar = _scalar(value)
            if math.isfinite(scalar):
                metric_name = (
                    name
                    if name.startswith(("overhead/", "performance/"))
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
            artifact_path = f"validation/step_{step:09d}/sample_{sample.index}"
            for relative_path in sample.paths:
                path = base / relative_path
                self.run.log_artifact(
                    path,
                    artifact_path=artifact_path,
                )


def start_run(config: TrainingConfig) -> TrackerRun:
    publish = config.studio_publish
    name = str(
        publish["run_name"]
        or publish["run_id"]
        or "finetune"
    )
    return start_mlflow_run(
        experiment="styletts2_finetune",
        name=name,
        config={
            "run_id": publish["run_id"],
            "finetune_job_id": publish["finetune_job_id"],
            "total_steps": config.total_steps,
            "batch_size": config.batch_size,
            "precision": config.precision,
            "distributed_processes": config.distributed_processes,
        },
    )


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)
