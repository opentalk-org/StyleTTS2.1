import time
from dataclasses import dataclass


@dataclass
class TrainingTelemetry:
    total_steps: int
    initial_step: int
    started_at: float
    items_processed: int = 0
    data_wait_seconds: float = 0.0
    compute_seconds: float = 0.0
    validation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    reporting_seconds: float = 0.0

    @classmethod
    def start(cls, total_steps: int, initial_step: int) -> "TrainingTelemetry":
        return cls(total_steps, initial_step, time.monotonic())

    def metrics(self, step: int) -> dict[str, float]:
        elapsed = time.monotonic() - self.started_at
        measured_steps = step - self.initial_step
        steps_per_second = measured_steps / elapsed
        eta_seconds = (self.total_steps - step) / steps_per_second
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
