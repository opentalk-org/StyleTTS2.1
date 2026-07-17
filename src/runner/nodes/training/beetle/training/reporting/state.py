from dataclasses import dataclass
from enum import StrEnum

from .metrics import MetricAccumulatorState
from .timing import TimingState


class ReportingCompletion(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass(frozen=True)
class ReportingState:
    mlflow_run_id: str | None
    timing: TimingState
    accumulator: MetricAccumulatorState
    last_reported_step: int
    last_validated_step: int
    pending_metric_operations: int
    pending_artifact_jobs: int
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

    @classmethod
    def initial(cls, mlflow_run_id: str | None = None) -> "ReportingState":
        return cls(
            mlflow_run_id,
            TimingState.initial(),
            MetricAccumulatorState.empty(),
            0,
            0,
            0,
            0,
            ReportingCompletion.ACTIVE,
        )

    def require_started(self) -> str:
        if self.mlflow_run_id is None:
            raise ValueError("checkpoint requires an active MLflow run ID")
        return self.mlflow_run_id
