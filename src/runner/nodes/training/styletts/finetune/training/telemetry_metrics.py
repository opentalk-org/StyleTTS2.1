import time
from collections import deque
from dataclasses import dataclass, field


THROUGHPUT_WINDOW_STEPS = 50


@dataclass(frozen=True)
class ThroughputSample:
    step: int
    elapsed_seconds: float
    audio_seconds_trained: float


@dataclass
class TrainingTelemetry:
    total_steps: int
    initial_step: int
    started_at: float
    items_processed: float = 0.0
    audio_seconds_trained: float = 0.0
    data_wait_seconds: float = 0.0
    compute_seconds: float = 0.0
    validation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    reporting_seconds: float = 0.0
    throughput_samples: deque[ThroughputSample] = field(
        default_factory=lambda: deque(maxlen=THROUGHPUT_WINDOW_STEPS + 1)
    )

    @classmethod
    def start(cls, total_steps: int, initial_step: int) -> "TrainingTelemetry":
        telemetry = cls(total_steps, initial_step, time.monotonic())
        telemetry.throughput_samples.append(
            ThroughputSample(initial_step, 0.0, 0.0)
        )
        return telemetry

    def metrics(self, step: int) -> dict[str, float]:
        elapsed = time.monotonic() - self.started_at
        measured_steps = step - self.initial_step
        steps_per_second = measured_steps / elapsed
        eta_seconds = (self.total_steps - step) / steps_per_second
        current_sample = ThroughputSample(
            step,
            elapsed,
            self.audio_seconds_trained,
        )
        self.throughput_samples.append(current_sample)
        window_start = self.throughput_samples[0]
        window_elapsed = elapsed - window_start.elapsed_seconds
        window_audio_seconds = (
            self.audio_seconds_trained - window_start.audio_seconds_trained
        )
        measured = (
            self.data_wait_seconds
            + self.compute_seconds
            + self.validation_seconds
            + self.checkpoint_seconds
            + self.reporting_seconds
        )
        overhead = {
            "data_wait": self.data_wait_seconds,
            "compute": self.compute_seconds,
            "validation": self.validation_seconds,
            "checkpoint": self.checkpoint_seconds,
            "reporting": self.reporting_seconds,
            "residual": max(elapsed - measured, 0.0),
        }
        metrics = {
            "performance/audio_seconds_per_second": window_audio_seconds / window_elapsed,
            "performance/items_per_second": self.items_processed / elapsed,
            "performance/steps_per_second": steps_per_second,
            "performance/elapsed_seconds": elapsed,
            "performance/eta_seconds": eta_seconds,
            "performance/eta_hours": eta_seconds / 3600,
        }
        metrics.update(
            {
                f"overhead/{name}_percent": 100 * value / elapsed
                for name, value in overhead.items()
            }
        )
        return metrics
