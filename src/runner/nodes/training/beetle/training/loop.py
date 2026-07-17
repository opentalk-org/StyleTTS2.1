import time
from dataclasses import dataclass, replace
from typing import Protocol

from ..data.prefetch import DataPipelineState
from ..data.records import BeetleBatch
from .callbacks import (
    CancellationRequested,
    TrainingCallbacks,
    TrainingMetric,
)
from .checkpoint import CheckpointManager, CheckpointPayload
from .loop_events import advance_sampler, announce, is_due, validate_metrics
from .reporting import (
    ForegroundCategory,
    ReportingState,
    StepObservationTracker,
    StepTimer,
)
from .state import LoopState, StageKind, TrainingPhase

_CHECKPOINT_MEDIA_TYPE = "application/vnd.beetle.checkpoint"


@dataclass(frozen=True)
class LoopIntervals:
    log_every_steps: int
    checkpoint_every_steps: int

    def __post_init__(self) -> None:
        values = (
            self.log_every_steps,
            self.checkpoint_every_steps,
        )
        if min(values) <= 0:
            raise ValueError("loop step intervals must be positive")


class TrainingPipeline(Protocol):
    def next_batch(self) -> BeetleBatch: ...

    def mark_consumed(self) -> None: ...

    def state_dict(self) -> DataPipelineState: ...


class StageTrainer(Protocol):
    stage: StageKind
    accumulation_steps: int
    trains_discriminator: bool
    intervals: LoopIntervals

    def loop_state(self) -> LoopState: ...

    def set_loop_state(self, state: LoopState) -> None: ...

    def discriminator_backward(
        self, batch: BeetleBatch
    ) -> tuple[TrainingMetric, ...]: ...

    def generator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]: ...

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]: ...

    def checkpoint_payload(
        self,
        loop: LoopState,
        sampler_state: DataPipelineState,
        reporting: ReportingState,
    ) -> CheckpointPayload: ...


def run_continuously(
    pipeline: TrainingPipeline,
    trainer: StageTrainer,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
) -> LoopState:
    timer = StepTimer()
    try:
        callbacks.check_cancel()
        while True:
            state = trainer.loop_state()
            if state.phase is TrainingPhase.GENERATOR_COMPLETE:
                _complete_accumulation(
                    trainer,
                    pipeline,
                    callbacks,
                    checkpoint_manager,
                    reporting,
                    timer,
                )
                continue
            if state.phase in (
                TrainingPhase.OPTIMIZER_COMPLETE,
                TrainingPhase.CHECKPOINTING,
            ):
                _complete_step_work(
                    trainer,
                    pipeline,
                    callbacks,
                    checkpoint_manager,
                    reporting,
                    timer,
                )
                continue
            _run_batch(
                pipeline,
                trainer,
                callbacks,
                checkpoint_manager,
                reporting,
                timer,
            )
    except CancellationRequested:
        elapsed, foreground = timer.snapshot()
        reporting.capture_partial_timing(elapsed, foreground)
        state = trainer.loop_state()
        payload = trainer.checkpoint_payload(
            state,
            pipeline.state_dict(),
            reporting.snapshot(),
        )
        path = checkpoint_manager.save(payload)
        callbacks.publish_artifact(path, _CHECKPOINT_MEDIA_TYPE)
        return state


def _run_batch(
    pipeline: TrainingPipeline,
    trainer: StageTrainer,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    timer: StepTimer,
) -> None:
    state = trainer.loop_state()
    resume_discriminator = state.phase in (
        TrainingPhase.DISCRIMINATOR_COMPLETE,
        TrainingPhase.GENERATOR_BACKWARD,
    )
    started_at = time.monotonic()
    batch = pipeline.next_batch()
    timer.record(ForegroundCategory.DATA_WAIT, started_at)
    if not resume_discriminator:
        state = replace(state, phase=TrainingPhase.BATCH_FETCHED)
        announce(trainer, callbacks, state, (), timer)
    metrics: tuple[TrainingMetric, ...] = ()
    if trainer.trains_discriminator and not resume_discriminator:
        state = replace(state, phase=TrainingPhase.DISCRIMINATOR_BACKWARD)
        announce(trainer, callbacks, state, (), timer)
        started_at = time.monotonic()
        discriminator_metrics = trainer.discriminator_backward(batch)
        timer.record(ForegroundCategory.COMPUTE, started_at)
        validate_metrics(discriminator_metrics)
        metrics += discriminator_metrics
        state = replace(state, phase=TrainingPhase.DISCRIMINATOR_COMPLETE)
        announce(trainer, callbacks, state, discriminator_metrics, timer)
    if trainer.loop_state().phase is not TrainingPhase.GENERATOR_BACKWARD:
        state = replace(state, phase=TrainingPhase.GENERATOR_BACKWARD)
        announce(trainer, callbacks, state, (), timer)
    started_at = time.monotonic()
    generator_metrics = trainer.generator_backward(batch)
    timer.record(ForegroundCategory.COMPUTE, started_at)
    validate_metrics(generator_metrics)
    metrics += generator_metrics
    pipeline.mark_consumed()
    reporting.add_microstep(len(batch.sample_keys), metrics)
    state = advance_sampler(state, pipeline.state_dict())
    state = replace(
        state,
        microstep=state.microstep + 1,
        phase=TrainingPhase.GENERATOR_COMPLETE,
    )
    announce(trainer, callbacks, state, generator_metrics, timer)
    _complete_accumulation(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting,
        timer,
    )


def _complete_accumulation(
    trainer: StageTrainer,
    pipeline: TrainingPipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager | None,
    reporting: StepObservationTracker,
    timer: StepTimer,
) -> None:
    state = trainer.loop_state()
    if state.microstep < trainer.accumulation_steps:
        ready = replace(state, phase=TrainingPhase.READY)
        trainer.set_loop_state(ready)
        callbacks.check_cancel()
        return
    if state.microstep > trainer.accumulation_steps:
        raise ValueError("microstep exceeds configured accumulation_steps")
    started_at = time.monotonic()
    step_metrics = trainer.optimizer_step(state.optimizer_step)
    timer.record(ForegroundCategory.COMPUTE, started_at)
    validate_metrics(step_metrics)
    state = replace(
        state,
        optimizer_step=state.optimizer_step + 1,
        microstep=0,
        phase=TrainingPhase.OPTIMIZER_COMPLETE,
    )
    elapsed, foreground = timer.complete()
    observation = reporting.complete_step(
        state.optimizer_step,
        step_metrics,
        elapsed,
        foreground,
    )
    report_metrics = (
        observation.metrics
        if is_due(state.optimizer_step, trainer.intervals.log_every_steps)
        else ()
    )
    announce(trainer, callbacks, state, report_metrics, timer)
    if checkpoint_manager is None:
        raise RuntimeError("checkpoint manager is required at optimizer boundaries")
    _complete_step_work(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting,
        timer,
    )


def _complete_step_work(
    trainer: StageTrainer,
    pipeline: TrainingPipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    timer: StepTimer,
) -> None:
    state = trainer.loop_state()
    checkpoint_due = is_due(
        state.optimizer_step, trainer.intervals.checkpoint_every_steps
    )
    if checkpoint_due:
        state = replace(state, phase=TrainingPhase.CHECKPOINTING)
        announce(trainer, callbacks, state, (), timer)
        complete = replace(state, phase=TrainingPhase.CHECKPOINT_COMPLETE)
        payload = trainer.checkpoint_payload(
            complete,
            pipeline.state_dict(),
            reporting.snapshot(),
        )
        path = checkpoint_manager.save(payload)
        callbacks.publish_artifact(path, _CHECKPOINT_MEDIA_TYPE)
        announce(trainer, callbacks, complete, (), timer)
        state = complete
    ready = replace(state, phase=TrainingPhase.READY)
    trainer.set_loop_state(ready)
    callbacks.check_cancel()
