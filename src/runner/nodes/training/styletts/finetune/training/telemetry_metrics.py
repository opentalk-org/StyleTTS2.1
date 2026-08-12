import time
from collections import deque
from dataclasses import dataclass, field


THROUGHPUT_WINDOW_STEPS = 50
THROUGHPUT_METRICS = (
    "audio_seconds_per_second",
    "items_per_second",
    "steps_per_second",
)
OVERHEAD_METRICS = (
    "data_wait",
    "compute",
    "validation",
    "checkpoint",
    "reporting",
    "residual",
)


@dataclass
class RollingAverage:
    window_size: int
    values: deque[float] = field(default_factory=deque)
    total: float = 0.0

    def step(self, value: float) -> float:
        if len(self.values) == self.window_size:
            self.total -= self.values.popleft()
        self.values.append(value)
        self.total += value
        return self.total / len(self.values)


@dataclass
class TrainingTelemetry:
    total_steps: int
    initial_step: int
    started_at: float
    last_step_at: float
    last_step: int
    items_processed: float = 0.0
    audio_seconds_trained: float = 0.0
    previous_items_processed: float = 0.0
    previous_audio_seconds_trained: float = 0.0
    data_wait_seconds: float = 0.0
    compute_seconds: float = 0.0
    validation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    reporting_seconds: float = 0.0
    previous_measured_seconds: dict[str, float] = field(
        default_factory=lambda: {
            name: 0.0 for name in OVERHEAD_METRICS if name != "residual"
        }
    )
    throughput_averages: dict[str, RollingAverage] = field(
        default_factory=lambda: {
            name: RollingAverage(THROUGHPUT_WINDOW_STEPS)
            for name in THROUGHPUT_METRICS
        }
    )
    overhead_averages: dict[str, RollingAverage] = field(
        default_factory=lambda: {
            name: RollingAverage(THROUGHPUT_WINDOW_STEPS)
            for name in (*OVERHEAD_METRICS, "elapsed")
        }
    )

    @classmethod
    def start(cls, total_steps: int, initial_step: int) -> "TrainingTelemetry":
        started_at = time.monotonic()
        return cls(
            total_steps=total_steps,
            initial_step=initial_step,
            started_at=started_at,
            last_step_at=started_at,
            last_step=initial_step,
        )

    def metrics(self, step: int) -> dict[str, float]:
        now = time.monotonic()
        elapsed = now - self.started_at
        step_elapsed = now - self.last_step_at
        rates = {
            "audio_seconds_per_second": (
                self.audio_seconds_trained - self.previous_audio_seconds_trained
            ) / step_elapsed,
            "items_per_second": (
                self.items_processed - self.previous_items_processed
            ) / step_elapsed,
            "steps_per_second": (step - self.last_step) / step_elapsed,
        }
        averages = {
            name: self.throughput_averages[name].step(value)
            for name, value in rates.items()
        }
        self.last_step_at = now
        self.last_step = step
        self.previous_items_processed = self.items_processed
        self.previous_audio_seconds_trained = self.audio_seconds_trained
        steps_per_second = averages["steps_per_second"]
        eta_seconds = (self.total_steps - step) / steps_per_second
        measured_totals = {
            name: getattr(self, f"{name}_seconds")
            for name in OVERHEAD_METRICS
            if name != "residual"
        }
        overhead = {
            name: total - self.previous_measured_seconds[name]
            for name, total in measured_totals.items()
        }
        overhead["residual"] = max(
            step_elapsed - sum(overhead.values()),
            0.0,
        )
        window_elapsed = self.overhead_averages["elapsed"].step(step_elapsed)
        window_overhead = {
            name: self.overhead_averages[name].step(value)
            for name, value in overhead.items()
        }
        self.previous_measured_seconds = measured_totals
        metrics = {
            **{
                f"performance/{name}": value
                for name, value in averages.items()
            },
            "performance/elapsed_seconds": elapsed,
            "performance/eta_seconds": eta_seconds,
            "performance/eta_hours": eta_seconds / 3600,
        }
        metrics.update(
            {
                f"overhead/{name}_percent": 100 * value / window_elapsed
                for name, value in window_overhead.items()
            }
        )
        return metrics
