import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMetric:
    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("training metric name must not be empty")


@dataclass(frozen=True)
class MetricAccumulatorState:
    names: tuple[str, ...]
    totals: tuple[float, ...]
    items: int
    microsteps: int

    def __post_init__(self) -> None:
        if len(self.names) != len(self.totals):
            raise ValueError("metric accumulator names and totals must align")
        if len(set(self.names)) != len(self.names):
            raise ValueError("metric accumulator names must be unique")
        if self.items < 0 or self.microsteps < 0:
            raise ValueError("metric accumulator counters must be non-negative")
        if not all(math.isfinite(value) for value in self.totals):
            raise ValueError("metric accumulator totals must be finite")

    @classmethod
    def empty(cls) -> "MetricAccumulatorState":
        return cls((), (), 0, 0)


@dataclass(frozen=True)
class CompletedMetrics:
    metrics: tuple[TrainingMetric, ...]
    items: int
    microsteps: int


class MetricAccumulator:
    def __init__(self, state: MetricAccumulatorState | None = None) -> None:
        self.state = state if state is not None else MetricAccumulatorState.empty()

    def add(self, items: int, metrics: tuple[TrainingMetric, ...]) -> None:
        if items <= 0:
            raise ValueError("completed microstep item count must be positive")
        names = tuple(metric.name for metric in metrics)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate training metric names: {names}")
        if self.state.microsteps > 0 and names != self.state.names:
            raise ValueError(
                f"microstep metric names changed: {names} != {self.state.names}"
            )
        totals = (
            tuple(metric.value for metric in metrics)
            if self.state.microsteps == 0
            else tuple(
                total + metric.value
                for total, metric in zip(self.state.totals, metrics, strict=True)
            )
        )
        self.state = MetricAccumulatorState(
            names,
            totals,
            self.state.items + items,
            self.state.microsteps + 1,
        )

    def complete(self) -> CompletedMetrics:
        if self.state.microsteps == 0:
            raise ValueError("cannot complete an empty metric accumulation")
        state = self.state
        metrics = tuple(
            TrainingMetric(name, total / state.microsteps)
            for name, total in zip(state.names, state.totals, strict=True)
        )
        self.state = MetricAccumulatorState.empty()
        return CompletedMetrics(metrics, state.items, state.microsteps)
