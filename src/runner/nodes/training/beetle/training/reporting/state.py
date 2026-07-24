from dataclasses import dataclass
from enum import StrEnum

from .metrics import MetricAccumulatorState, TrainingMetric
from .timing import TimingState


class ReportingCompletion(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass(frozen=True)
class PendingStepState:
    optimizer_step: int
    optimizer_metrics: tuple[TrainingMetric, ...]

    def __post_init__(self) -> None:
        if self.optimizer_step <= 0:
            raise ValueError("pending optimizer step must be positive")
        names = tuple(metric.name for metric in self.optimizer_metrics)
        if not self.optimizer_metrics or len(set(names)) != len(names):
            raise ValueError("pending optimizer metrics must be nonempty and unique")


@dataclass(frozen=True)
class ReportingState:
    mlflow_run_id: str | None
    timing: TimingState
    accumulator: MetricAccumulatorState
    last_reported_step: int
    last_validated_step: int
    pending_metric_operations: int
    pending_artifact_jobs: int
    pending_step: PendingStepState | None
    completion: ReportingCompletion

    def __post_init__(self) -> None:
        counters = (
            self.last_reported_step,
            self.last_validated_step,
            self.pending_metric_operations,
            self.pending_artifact_jobs,
        )
        if min(counters) < 0:
            raise ValueError("reporting counters must be non-negative")
        if self.mlflow_run_id is not None and not self.mlflow_run_id:
            raise ValueError("MLflow run ID must not be empty")
        if (
            self.pending_step is not None
            and self.pending_step.optimizer_step <= self.last_reported_step
        ):
            raise ValueError("pending optimizer step must follow the reported step")

    @classmethod
    def initial(cls, mlflow_run_id: str) -> "ReportingState":
        return cls(
            mlflow_run_id,
            TimingState.initial(),
            MetricAccumulatorState.empty(),
            0,
            0,
            0,
            0,
            None,
            ReportingCompletion.ACTIVE,
        )

    def require_started(self) -> str:
        if self.mlflow_run_id is None:
            raise ValueError("checkpoint requires an active MLflow run ID")
        return self.mlflow_run_id
