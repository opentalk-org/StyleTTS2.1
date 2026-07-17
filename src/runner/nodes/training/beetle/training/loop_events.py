import math
import time
from dataclasses import replace
from typing import Protocol

from ..data.prefetch import DataPipelineState
from .callbacks import ProgressEvent, TrainingCallbacks
from .reporting import (
    ForegroundCategory,
    StepTimer,
    TrainingMetric,
)
from .state import LoopState


class LoopStateOwner(Protocol):
    def set_loop_state(self, state: LoopState) -> None: ...


def advance_sampler(state: LoopState, sampler: DataPipelineState) -> LoopState:
    planner = sampler.planner
    return replace(
        state,
        sampler_cursor=planner.batch_index,
        cycle=max(planner.sentence.cycle_index, planner.mid_sentence.cycle_index),
        batch_index=planner.batch_index,
    )


def announce(
    owner: LoopStateOwner,
    callbacks: TrainingCallbacks,
    state: LoopState,
    metrics: tuple[TrainingMetric, ...],
    timer: StepTimer,
) -> None:
    started_at = time.monotonic()
    try:
        owner.set_loop_state(state)
        callbacks.report_progress(
            ProgressEvent(
                state.stage,
                state.optimizer_step,
                state.microstep,
                state.phase,
                metrics,
            )
        )
        callbacks.check_cancel()
    finally:
        timer.record(ForegroundCategory.REPORTING, started_at)


def validate_metrics(metrics: tuple[TrainingMetric, ...]) -> None:
    names = tuple(metric.name for metric in metrics)
    if len(set(names)) != len(names):
        raise ValueError(f"training metric names must be unique: {names}")
    invalid = tuple(
        metric.name for metric in metrics if not math.isfinite(metric.value)
    )
    if invalid:
        raise FloatingPointError(f"non-finite training metrics: {invalid}")


def is_due(optimizer_step: int, interval: int) -> bool:
    return optimizer_step > 0 and optimizer_step % interval == 0
