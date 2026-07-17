import math
import time
from dataclasses import dataclass, replace
from enum import StrEnum


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
    partial_elapsed_seconds: float
    partial_foreground: ForegroundDurations

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
        if (
            not math.isfinite(self.partial_elapsed_seconds)
            or self.partial_elapsed_seconds < 0
        ):
            raise ValueError("partial elapsed time must be finite and non-negative")
        if not math.isclose(
            self.partial_foreground.total,
            self.partial_elapsed_seconds,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("partial foreground must partition partial elapsed time")

    @classmethod
    def initial(cls) -> "TimingState":
        return cls(
            0,
            0,
            0,
            0.0,
            ForegroundDurations(),
            0.0,
            ForegroundDurations(),
        )

    def with_partial(
        self,
        elapsed_seconds: float,
        foreground: ForegroundDurations,
    ) -> "TimingState":
        if not math.isclose(
            foreground.total,
            elapsed_seconds,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("partial foreground must partition elapsed time")
        return replace(
            self,
            partial_elapsed_seconds=self.partial_elapsed_seconds + elapsed_seconds,
            partial_foreground=self.partial_foreground.plus(foreground),
        )

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
        combined_elapsed = self.partial_elapsed_seconds + elapsed_seconds
        combined_foreground = self.partial_foreground.plus(foreground)
        if self.last_completed_step == 0:
            return TimingState(
                optimizer_step,
                0,
                0,
                0.0,
                ForegroundDurations(),
                0.0,
                ForegroundDurations(),
            )
        return TimingState(
            optimizer_step,
            self.measured_steps + 1,
            self.measured_items + items,
            self.elapsed_seconds + combined_elapsed,
            self.foreground.plus(combined_foreground),
            0.0,
            ForegroundDurations(),
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


class ForegroundCategory(StrEnum):
    DATA_WAIT = "data_wait"
    COMPUTE = "compute"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    REPORTING = "reporting"


class StepTimer:
    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._foreground = ForegroundDurations()

    def record(self, category: ForegroundCategory, started_at: float) -> None:
        duration = time.monotonic() - started_at
        if duration < 0:
            raise ValueError("foreground timer moved backwards")
        match category:
            case ForegroundCategory.DATA_WAIT:
                updated = replace(
                    self._foreground,
                    data_wait=self._foreground.data_wait + duration,
                )
            case ForegroundCategory.COMPUTE:
                updated = replace(
                    self._foreground,
                    compute=self._foreground.compute + duration,
                )
            case ForegroundCategory.VALIDATION:
                updated = replace(
                    self._foreground,
                    validation=self._foreground.validation + duration,
                )
            case ForegroundCategory.CHECKPOINT:
                updated = replace(
                    self._foreground,
                    checkpoint=self._foreground.checkpoint + duration,
                )
            case ForegroundCategory.REPORTING:
                updated = replace(
                    self._foreground,
                    reporting=self._foreground.reporting + duration,
                )
        self._foreground = updated

    def snapshot(self) -> tuple[float, ForegroundDurations]:
        return self._snapshot(time.monotonic())

    def complete(self) -> tuple[float, ForegroundDurations]:
        now = time.monotonic()
        snapshot = self._snapshot(now)
        self._started_at = now
        self._foreground = ForegroundDurations()
        return snapshot

    def _snapshot(self, now: float) -> tuple[float, ForegroundDurations]:
        elapsed = now - self._started_at
        residual = elapsed - self._foreground.total
        if residual < -1e-9:
            raise ValueError("foreground timings overlap")
        durations = replace(self._foreground, residual=max(residual, 0.0))
        return durations.total, durations
