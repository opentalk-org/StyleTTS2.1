from __future__ import annotations

from collections import deque


def overhead_percent(
    examples_per_second: float,
    batch_size: int,
    steps_per_second: float,
) -> float:
    compute_steps_per_second = examples_per_second / batch_size
    overhead = 100 * (1 - steps_per_second / compute_steps_per_second)
    return min(max(overhead, 0.0), 100.0)


class TrainingProgressEstimator:
    """Estimate throughput and ETA from recent completed-step timestamps."""

    def __init__(self, total_steps: int, window_steps: int, warmup_intervals: int) -> None:
        self.total_steps = total_steps
        self.warmup_intervals = warmup_intervals
        self.completions: deque[tuple[int, float]] = deque(maxlen=window_steps + 1)

    def resume(self, now: float) -> None:
        """Move the rolling clock past validation and epoch-start loading pauses."""
        assert self.completions, "cannot resume progress before the first completed step"
        gap = now - self.completions[-1][1]
        self.completions = deque(
            ((step, completed_at + gap) for step, completed_at in self.completions),
            maxlen=self.completions.maxlen,
        )

    def update(self, global_step: int, now: float) -> dict[str, float]:
        self.completions.append((global_step, now))
        first_step, first_time = self.completions[0]
        completed_intervals = global_step - first_step
        if completed_intervals < self.warmup_intervals:
            return {}
        elapsed = now - first_time
        steps_per_second = completed_intervals / elapsed
        eta_seconds = max(self.total_steps - global_step, 0) / steps_per_second
        return {
            "steps_per_second": steps_per_second,
            "eta_seconds": eta_seconds,
            "eta_hours": eta_seconds / 3600,
        }
