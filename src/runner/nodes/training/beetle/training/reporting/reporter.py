from dataclasses import replace
from typing import Protocol

from .metrics import TrainingMetric
from .mlflow import MAX_PENDING_OPERATIONS, MlflowSession
from .observations import StepObservation
from .state import ReportingCompletion, ReportingState

MAX_PENDING_ARTIFACT_JOBS = 16


class SystemSampler(Protocol):
    def sample(self) -> tuple[TrainingMetric, ...]: ...


class TrainingReporter:
    def __init__(
        self,
        session: MlflowSession,
        total_steps: int,
        system: SystemSampler,
    ) -> None:
        self.session = session
        self.total_steps = total_steps
        self.system = system

    def publish(
        self,
        observation: StepObservation,
        state: ReportingState,
        validation_metrics: tuple[TrainingMetric, ...],
    ) -> ReportingState:
        metrics = (
            *tuple(
                TrainingMetric(f"train/{metric.name}", metric.value)
                for metric in observation.losses
            ),
            *observation.optimizer_metrics,
            *self._performance_metrics(state),
            *self._overhead_metrics(state),
            *self.system.sample(),
            *validation_metrics,
        )
        self.session.submit(metrics, observation.optimizer_step)
        return replace(
            state,
            last_reported_step=observation.optimizer_step,
            pending_metric_operations=self.session.pending_count,
        )

    def flush(self) -> None:
        self.session.flush()

    def finish(self, state: ReportingState) -> ReportingState:
        self.session.finish()
        return replace(
            state,
            pending_metric_operations=0,
            completion=ReportingCompletion.FINISHED,
        )

    def fail(self) -> None:
        self.session.fail()

    def _performance_metrics(
        self,
        state: ReportingState,
    ) -> tuple[TrainingMetric, ...]:
        timing = state.timing
        if timing.measured_steps == 0:
            return ()
        snapshot = timing.snapshot(self.total_steps)
        return (
            TrainingMetric(
                "performance/items_per_second",
                snapshot.items_per_second,
            ),
            TrainingMetric(
                "performance/steps_per_second",
                snapshot.steps_per_second,
            ),
            TrainingMetric("performance/elapsed_seconds", timing.elapsed_seconds),
            TrainingMetric("performance/eta_seconds", snapshot.eta_seconds),
            TrainingMetric("performance/eta_hours", snapshot.eta_hours),
        )

    def _overhead_metrics(
        self,
        state: ReportingState,
    ) -> tuple[TrainingMetric, ...]:
        timing = state.timing
        if timing.measured_steps == 0:
            return ()
        foreground = timing.foreground
        elapsed = timing.elapsed_seconds
        categories = (
            ("data_wait", foreground.data_wait),
            ("compute", foreground.compute),
            ("validation", foreground.validation),
            ("checkpoint", foreground.checkpoint),
            ("reporting", foreground.reporting),
            ("residual", foreground.residual),
        )
        percentages = tuple(
            TrainingMetric(f"overhead/{name}_percent", 100 * value / elapsed)
            for name, value in categories
        )
        return (
            *percentages,
            TrainingMetric(
                "overhead/pending_metric_operations",
                float(self.session.pending_count),
            ),
            TrainingMetric(
                "overhead/metric_queue_utilization_percent",
                100 * self.session.pending_count / MAX_PENDING_OPERATIONS,
            ),
            TrainingMetric(
                "overhead/pending_artifact_jobs",
                float(state.pending_artifact_jobs),
            ),
            TrainingMetric(
                "overhead/artifact_queue_utilization_percent",
                100 * state.pending_artifact_jobs / MAX_PENDING_ARTIFACT_JOBS,
            ),
        )
