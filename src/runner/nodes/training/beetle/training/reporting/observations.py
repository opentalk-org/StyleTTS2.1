from dataclasses import dataclass, replace

from .metrics import MetricAccumulator, TrainingMetric
from .state import ReportingState
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

    def complete_step(
        self,
        optimizer_step: int,
        optimizer_metrics: tuple[TrainingMetric, ...],
        elapsed_seconds: float,
        foreground: ForegroundDurations,
    ) -> StepObservation:
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
        )
        return StepObservation(
            optimizer_step,
            completed.metrics,
            optimizer_metrics,
            completed.items,
            elapsed_seconds,
            foreground,
        )

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
