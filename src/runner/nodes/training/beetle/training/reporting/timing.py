import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForegroundDurations:
    data_wait: float = 0.0
    compute: float = 0.0
    validation: float = 0.0
    checkpoint: float = 0.0
    reporting: float = 0.0
    residual: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.data_wait,
            self.compute,
            self.validation,
            self.checkpoint,
            self.reporting,
            self.residual,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("foreground durations must be finite and non-negative")

    @property
    def total(self) -> float:
        return sum(
            (
                self.data_wait,
                self.compute,
                self.validation,
                self.checkpoint,
                self.reporting,
                self.residual,
            )
        )

    def plus(self, other: "ForegroundDurations") -> "ForegroundDurations":
        return ForegroundDurations(
            self.data_wait + other.data_wait,
            self.compute + other.compute,
            self.validation + other.validation,
            self.checkpoint + other.checkpoint,
            self.reporting + other.reporting,
            self.residual + other.residual,
        )


@dataclass(frozen=True)
class TimingSnapshot:
    steps_per_second: float
    items_per_second: float
    eta_seconds: float
    eta_hours: float


@dataclass(frozen=True)
class TimingState:
    last_completed_step: int
    measured_steps: int
    measured_items: int
    elapsed_seconds: float
    foreground: ForegroundDurations

    def __post_init__(self) -> None:
        counters = (
            self.last_completed_step,
            self.measured_steps,
            self.measured_items,
        )
        if min(counters) < 0:
            raise ValueError("timing counters must be non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed time must be finite and non-negative")

    @classmethod
    def initial(cls) -> "TimingState":
        return cls(0, 0, 0, 0.0, ForegroundDurations())

    def record_step(
        self,
        optimizer_step: int,
        items: int,
        elapsed_seconds: float,
        foreground: ForegroundDurations,
    ) -> "TimingState":
        if optimizer_step != self.last_completed_step + 1:
            raise ValueError("completed optimizer step must advance by exactly one")
        if items <= 0:
            raise ValueError("completed optimizer step item count must be positive")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("completed optimizer step elapsed time must be positive")
        if not math.isclose(foreground.total, elapsed_seconds, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("foreground durations must partition elapsed time")
        if self.last_completed_step == 0:
            return TimingState(optimizer_step, 0, 0, 0.0, ForegroundDurations())
        return TimingState(
            optimizer_step,
            self.measured_steps + 1,
            self.measured_items + items,
            self.elapsed_seconds + elapsed_seconds,
            self.foreground.plus(foreground),
        )

    def snapshot(self, total_steps: int) -> TimingSnapshot:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.measured_steps == 0 or self.elapsed_seconds == 0:
            raise ValueError("timing rates require a measured optimizer step")
        average = self.elapsed_seconds / self.measured_steps
        eta = (
            0.0
            if self.last_completed_step >= total_steps
            else average * total_steps - self.elapsed_seconds
        )
        return TimingSnapshot(
            self.measured_steps / self.elapsed_seconds,
            self.measured_items / self.elapsed_seconds,
            eta,
            eta / 3600,
        )
