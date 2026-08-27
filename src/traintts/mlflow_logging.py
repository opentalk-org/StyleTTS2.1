import logging
import math
from pathlib import Path

import torch

from .tracking import (
    LocalTracker,
    TrackerRun,
    # mlflow disabled for now
    # resume_mlflow_run,
    # start_mlflow_run,
)
from .val_sample_export import ValidationSampleArtifacts
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
            artifact_path = (
                f"validation/step_{step:09d}/{sample.mode}/sample_{sample.index}"
            )
            for relative_path in sample.paths:
                path = base / relative_path
                self.run.log_artifact(
                    path,
                    artifact_path=artifact_path,
                )


def start_run(config: TrainingConfig) -> TrackerRun:
    logging.getLogger(__name__).info(
        "logging metrics to %s/metrics.jsonl",
        config.log_dir,
    )
    return LocalTracker(config.log_dir)
    # mlflow disabled for now
    # if not os.environ.get("MLFLOW_TRACKING_URI"):
    #     return LocalTracker(config.log_dir)
    # publish = config.studio_publish
    # resume_run_id = str(publish["mlflow_run_id"])
    # if resume_run_id:
    #     return resume_mlflow_run(resume_run_id)
    # name = str(
    #     publish["run_name"]
    #     or publish["run_id"]
    #     or "finetune"
    # )
    # return start_mlflow_run(
    #     experiment="styletts2_finetune",
    #     name=name,
    #     config={
    #         "run_id": publish["run_id"],
    #         "finetune_job_id": publish["finetune_job_id"],
    #         "total_steps": config.total_steps,
    #         "stage_max_audio_seconds": ",".join(
    #             str(stage.max_audio_seconds) for stage in config.training_stages
    #         ),
    #         "precision": config.precision,
    #         "distributed_processes": config.distributed_processes,
    #     },
    # )


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)
