from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMetric:
    name: str
    value: float


@dataclass(frozen=True)
class MetricAccumulatorState:
    names: tuple[str, ...]
    totals: tuple[float, ...]
    items: int
    microsteps: int

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
        names = tuple(metric.name for metric in metrics)
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
        state = self.state
        metrics = tuple(
            TrainingMetric(name, total / state.microsteps)
            for name, total in zip(state.names, state.totals, strict=True)
        )
        self.state = MetricAccumulatorState.empty()
        return CompletedMetrics(metrics, state.items, state.microsteps)
