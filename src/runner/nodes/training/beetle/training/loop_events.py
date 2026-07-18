import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from ..data.prefetch import DataPipelineState
from .callbacks import (
    ProgressEvent,
    TrainingCallbacks,
    TrainingMetric,
    is_due,
    report_only,
)
from .checkpoint import CheckpointManager, CheckpointPayload
from .reporting import (
    ForegroundCategory,
    ReportingCompletion,
    ReportingState,
    StepObservation,
    StepObservationTracker,
    StepTimer,
)
from .state import LoopState, TrainingPhase

_CHECKPOINT_MEDIA_TYPE = "application/vnd.beetle.checkpoint"


class LoopStateOwner(Protocol):
    def set_loop_state(self, state: LoopState) -> None: ...


class LifecyclePipeline(Protocol):
    def state_dict(self) -> DataPipelineState: ...


class LifecycleTrainer(LoopStateOwner, Protocol):
    def loop_state(self) -> LoopState: ...

    def checkpoint_payload(
        self,
        loop: LoopState,
        sampler_state: DataPipelineState,
        reporting: ReportingState,
    ) -> CheckpointPayload: ...


class StepReporter(Protocol):
    def publish(
        self,
        observation: StepObservation,
        state: ReportingState,
        validation_metrics: tuple[TrainingMetric, ...],
    ) -> ReportingState: ...

    def flush(self) -> None: ...

    def finish(self, state: ReportingState) -> ReportingState: ...

    def fail(self) -> None: ...


class StepValidator(Protocol):
    def run(self, step: int) -> tuple[TrainingMetric, ...]: ...


@dataclass(frozen=True)
class TrainingLifecycle:
    total_steps: int
    validation_every_steps: int
    reporter: StepReporter
    validator: StepValidator

    def __post_init__(self) -> None:
        if self.total_steps <= 0 or self.validation_every_steps <= 0:
            raise ValueError("lifecycle step limits must be positive")


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


def finish_run(
    trainer: LifecycleTrainer,
    pipeline: LifecyclePipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
    timer: StepTimer,
) -> LoopState:
    state = trainer.loop_state()
    snapshot = reporting.snapshot()
    if snapshot.completion is ReportingCompletion.FINISHED:
        return state
    if snapshot.last_reported_step != lifecycle.total_steps:
        raise ValueError("final optimizer step has not been reported")
    if snapshot.last_validated_step != lifecycle.total_steps:
        raise ValueError("final optimizer step has not been validated")
    lifecycle.reporter.flush()
    reporting.mark_flushed()
    reporting.update(lifecycle.reporter.finish(reporting.snapshot()))
    return save_checkpoint(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting.snapshot(),
        state,
        timer,
    )


def save_checkpoint(
    trainer: LifecycleTrainer,
    pipeline: LifecyclePipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: ReportingState,
    state: LoopState,
    timer: StepTimer,
) -> LoopState:
    started_at = time.monotonic()
    checkpointing = replace(state, phase=TrainingPhase.CHECKPOINTING)
    trainer.set_loop_state(checkpointing)
    complete = replace(checkpointing, phase=TrainingPhase.CHECKPOINT_COMPLETE)
    payload = trainer.checkpoint_payload(
        complete,
        pipeline.state_dict(),
        reporting,
    )
    path: Path = checkpoint_manager.save(payload)
    callbacks.publish_artifact(path, _CHECKPOINT_MEDIA_TYPE)
    trainer.set_loop_state(complete)
    timer.record(ForegroundCategory.CHECKPOINT, started_at)
    return complete


def save_emergency_checkpoint(
    trainer: LifecycleTrainer,
    pipeline: LifecyclePipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: ReportingState,
    state: LoopState,
    timer: StepTimer,
) -> Path:
    started_at = time.monotonic()
    payload = trainer.checkpoint_payload(
        state,
        pipeline.state_dict(),
        reporting,
    )
    path = checkpoint_manager.save(payload)
    callbacks.publish_artifact(path, _CHECKPOINT_MEDIA_TYPE)
    trainer.set_loop_state(state)
    timer.record(ForegroundCategory.CHECKPOINT, started_at)
    return path


def cancel_run(
    trainer: LifecycleTrainer,
    pipeline: LifecyclePipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
    timer: StepTimer,
) -> LoopState:
    flush_error: Exception | None = None
    try:
        lifecycle.reporter.flush()
        reporting.mark_flushed()
    except Exception as error:
        flush_error = error
    state = trainer.loop_state()
    save_emergency_checkpoint(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting.snapshot(),
        state,
        timer,
    )
    if flush_error is not None:
        lifecycle.reporter.fail()
        raise flush_error
    return state


def complete_step_work(
    trainer: LifecycleTrainer,
    pipeline: LifecyclePipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
    timer: StepTimer,
    log_every_steps: int,
    checkpoint_every_steps: int,
) -> None:
    state = trainer.loop_state()
    pending = reporting.snapshot().pending_step
    if pending is None or pending.optimizer_step != state.optimizer_step:
        raise ValueError("optimizer boundary has no matching pending observation")
    validation_metrics: tuple[TrainingMetric, ...] = ()
    validation_due = is_due(
        state.optimizer_step,
        lifecycle.validation_every_steps,
    ) or state.optimizer_step == lifecycle.total_steps
    if validation_due:
        started_at = time.monotonic()
        validation_metrics = lifecycle.validator.run(state.optimizer_step)
        timer.record(ForegroundCategory.VALIDATION, started_at)
        reporting.update(
            replace(
                reporting.snapshot(),
                last_validated_step=state.optimizer_step,
            )
        )
    elapsed, foreground = timer.complete()
    observation = reporting.complete_step(
        state.optimizer_step,
        elapsed,
        foreground,
    )
    reporting.update(
        lifecycle.reporter.publish(
            observation,
            reporting.snapshot(),
            validation_metrics,
        )
    )
    report_metrics = (
        observation.metrics
        if is_due(state.optimizer_step, log_every_steps)
        else ()
    )
    trainer.set_loop_state(replace(state, phase=TrainingPhase.READY))
    report_only(callbacks, state, report_metrics, timer)
    checkpoint_due = is_due(
        state.optimizer_step,
        checkpoint_every_steps,
    ) or state.optimizer_step == lifecycle.total_steps
    if checkpoint_due:
        lifecycle.reporter.flush()
        reporting.mark_flushed()
        state = save_checkpoint(
            trainer,
            pipeline,
            callbacks,
            checkpoint_manager,
            reporting.snapshot(),
            state,
            timer,
        )
    trainer.set_loop_state(replace(state, phase=TrainingPhase.READY))
    callbacks.check_cancel()
