import math
from pathlib import Path

import torch

from runner.nodes.training.common.mlflow_run import (
    TrackerRun,
    start_mlflow_run,
)

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
        for name, value in metrics.items():
            scalar = _scalar(value)
            if math.isfinite(scalar):
                self.run.track(scalar, name=f"train/{name}", step=step)
        progress = max(
            1.0,
            min(99.0, 1.0 + 98.0 * step / self.total_steps),
        )
        self.run.track(progress, name="job_progress", step=step)

    def log_validation(
        self,
        step: int,
        metrics: dict[str, torch.Tensor | float],
        samples: list[dict[str, str]],
        log_dir: str | Path,
    ) -> None:
        for name, value in metrics.items():
            scalar = _scalar(value)
            if math.isfinite(scalar):
                self.run.track(scalar, name=f"val/{name}", step=step)
        self.run.track(
            float(len(samples)),
            name="val/sample_rows",
            step=step,
        )
        base = Path(log_dir)
        for sample in samples:
            path = base / sample["path"]
            if path.is_file():
                self.run.log_artifact(
                    path,
                    artifact_path=(
                        f"validation/step_{step:09d}/"
                        f"sample_{sample['index']}/{sample['role']}"
                    ),
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
