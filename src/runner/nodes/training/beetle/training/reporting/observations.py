from dataclasses import dataclass, replace

from .metrics import MetricAccumulator, TrainingMetric
from .state import PendingStepState, ReportingState
from .timing import ForegroundDurations


@dataclass(frozen=True)
class StepObservation:
    optimizer_step: int
    losses: tuple[TrainingMetric, ...]
    optimizer_metrics: tuple[TrainingMetric, ...]
    items: int
    elapsed_seconds: float
    foreground: ForegroundDurations

    @property
    def metrics(self) -> tuple[TrainingMetric, ...]:
        return (*self.losses, *self.optimizer_metrics)


class StepObservationTracker:
    def __init__(self, state: ReportingState) -> None:
        self._state = state
        self._accumulator = MetricAccumulator(state.accumulator)

    def add_microstep(
        self,
        items: int,
        losses: tuple[TrainingMetric, ...],
    ) -> None:
        self._accumulator.add(items, losses)
        self._state = replace(self._state, accumulator=self._accumulator.state)

    def discard_accumulation(self) -> None:
        self._accumulator = MetricAccumulator()
        self._state = replace(self._state, accumulator=self._accumulator.state)

    def complete_step(
        self,
        optimizer_step: int,
        elapsed_seconds: float,
        foreground: ForegroundDurations,
    ) -> StepObservation:
        pending = self._state.pending_step
        completed = self._accumulator.complete()
        timing = self._state.timing.record_step(
            optimizer_step,
            completed.items,
            elapsed_seconds,
            foreground,
        )
        self._state = replace(
            self._state,
            timing=timing,
            accumulator=self._accumulator.state,
            pending_step=None,
        )
        return StepObservation(
            optimizer_step,
            completed.metrics,
            pending.optimizer_metrics,
            completed.items,
            elapsed_seconds,
            foreground,
        )

    def begin_step(
        self,
        optimizer_step: int,
        optimizer_metrics: tuple[TrainingMetric, ...],
    ) -> None:
        self._state = replace(
            self._state,
            pending_step=PendingStepState(optimizer_step, optimizer_metrics),
        )

    def update(self, state: ReportingState) -> None:
        self._state = state

    def mark_flushed(self) -> None:
        self._state = replace(self._state, pending_metric_operations=0)

    def capture_partial_timing(
        self,
        elapsed_seconds: float,
        foreground: ForegroundDurations,
    ) -> None:
        self._state = replace(
            self._state,
            timing=self._state.timing.with_partial(elapsed_seconds, foreground),
        )

    def snapshot(self) -> ReportingState:
        return self._state
