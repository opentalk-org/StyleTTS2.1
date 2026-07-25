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
        return self.mlflow_run_id
