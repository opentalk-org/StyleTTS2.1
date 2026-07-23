import time
from dataclasses import dataclass, replace
from typing import Protocol

from ..data.prefetch import DataPipelineState, TrainingBatch
from .callbacks import (
    CancellationRequested,
    TrainingCallbacks,
    TrainingMetric,
    validate_metrics,
)
from .checkpoint import CheckpointManager, CheckpointPayload
from .loop_events import (
    TrainingLifecycle,
    advance_sampler,
    announce,
    cancel_run,
    complete_step_work,
    finish_run,
    skip_batch,
    skip_optimizer_step,
)
from .reporting import (
    ForegroundCategory,
    ReportingState,
    StepObservationTracker,
    StepTimer,
)
from .state import LoopState, TrainingPhase


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
    def next_batch(self) -> TrainingBatch: ...

    def mark_consumed(self) -> None: ...

    def state_dict(self) -> DataPipelineState: ...


class TrainingController(Protocol):
    accumulation_steps: int
    trains_discriminator: bool
    intervals: LoopIntervals
    world_size: int

    def loop_state(self) -> LoopState: ...

    def set_loop_state(self, state: LoopState) -> None: ...

    def discriminator_backward(
        self, batch: TrainingBatch
    ) -> tuple[TrainingMetric, ...]: ...

    def generator_backward(
        self, batch: TrainingBatch
    ) -> tuple[TrainingMetric, ...]: ...

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]: ...

    def discard_step(self) -> None: ...

    def reduce_metrics(
        self, metrics: tuple[TrainingMetric, ...]
    ) -> tuple[TrainingMetric, ...]: ...

    def checkpoint_payload(
        self,
        loop: LoopState,
        sampler_state: DataPipelineState,
        reporting: ReportingState,
    ) -> CheckpointPayload: ...


def run_continuously(
    pipeline: TrainingPipeline,
    trainer: TrainingController,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
) -> LoopState:
    timer = StepTimer()
    try:
        callbacks.check_cancel()
        while True:
            state = trainer.loop_state()
            if state.optimizer_step > lifecycle.total_steps:
                raise ValueError("optimizer step exceeds configured total_steps")
            if (
                state.optimizer_step == lifecycle.total_steps
                and reporting.snapshot().pending_step is None
            ):
                return finish_run(
                    trainer,
                    pipeline,
                    callbacks,
                    checkpoint_manager,
                    reporting,
                    lifecycle,
                    timer,
                )
            if state.phase is TrainingPhase.GENERATOR_COMPLETE:
                _complete_accumulation(
                    trainer,
                    pipeline,
                    callbacks,
                    checkpoint_manager,
                    reporting,
                    lifecycle,
                    timer,
                )
                continue
            if state.phase is TrainingPhase.OPTIMIZER_COMPLETE:
                complete_step_work(
                    trainer,
                    pipeline,
                    callbacks,
                    checkpoint_manager,
                    reporting,
                    lifecycle,
                    timer,
                    trainer.intervals.log_every_steps,
                    trainer.intervals.checkpoint_every_steps,
                )
                continue
            _run_batch(
                pipeline,
                trainer,
                callbacks,
                checkpoint_manager,
                reporting,
                lifecycle,
                timer,
            )
    except CancellationRequested:
        elapsed, foreground = timer.snapshot()
        reporting.capture_partial_timing(elapsed, foreground)
        return cancel_run(
            trainer,
            reporting,
            lifecycle,
        )
    except Exception as error:
        elapsed, foreground = timer.snapshot()
        reporting.capture_partial_timing(elapsed, foreground)
        if trainer.world_size > 1:
            try:
                lifecycle.reporter.fail()
            except Exception:
                pass
            raise error
        try:
            lifecycle.reporter.fail()
        except Exception:
            pass
        raise error


def _run_batch(
    pipeline: TrainingPipeline,
    trainer: TrainingController,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
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
    metrics = state.discriminator_metrics if resume_discriminator else ()
    if trainer.trains_discriminator and not resume_discriminator:
        state = replace(state, phase=TrainingPhase.DISCRIMINATOR_BACKWARD)
        announce(trainer, callbacks, state, (), timer)
        started_at = time.monotonic()
        try:
            discriminator_metrics = trainer.reduce_metrics(
                trainer.discriminator_backward(batch)
            )
            timer.record(ForegroundCategory.COMPUTE, started_at)
            validate_metrics(discriminator_metrics)
        except FloatingPointError:
            skip_batch(trainer, pipeline, callbacks, reporting, timer)
            return
        metrics += discriminator_metrics
        state = replace(
            state,
            phase=TrainingPhase.DISCRIMINATOR_COMPLETE,
            discriminator_metrics=discriminator_metrics,
        )
        announce(trainer, callbacks, state, discriminator_metrics, timer)
    if trainer.loop_state().phase is not TrainingPhase.GENERATOR_BACKWARD:
        state = replace(state, phase=TrainingPhase.GENERATOR_BACKWARD)
        announce(trainer, callbacks, state, (), timer)
    started_at = time.monotonic()
    try:
        generator_metrics = trainer.reduce_metrics(trainer.generator_backward(batch))
        timer.record(ForegroundCategory.COMPUTE, started_at)
        validate_metrics(generator_metrics)
    except FloatingPointError:
        skip_batch(trainer, pipeline, callbacks, reporting, timer)
        return
    metrics += generator_metrics
    pipeline.mark_consumed()
    reporting.add_microstep(
        len(batch.sample_keys) * trainer.world_size,
        metrics,
    )
    state = advance_sampler(state, pipeline.state_dict())
    state = replace(
        state,
        microstep=state.microstep + 1,
        phase=TrainingPhase.GENERATOR_COMPLETE,
        discriminator_metrics=(),
    )
    announce(trainer, callbacks, state, generator_metrics, timer)
    _complete_accumulation(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting,
        lifecycle,
        timer,
    )


def _complete_accumulation(
    trainer: TrainingController,
    pipeline: TrainingPipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager | None,
    reporting: StepObservationTracker,
    lifecycle: TrainingLifecycle,
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
    try:
        step_metrics = trainer.reduce_metrics(
            trainer.optimizer_step(state.optimizer_step)
        )
    except FloatingPointError:
        timer.record(ForegroundCategory.COMPUTE, started_at)
        skip_optimizer_step(
            trainer,
            pipeline,
            callbacks,
            reporting,
            timer,
        )
        return
    timer.record(ForegroundCategory.COMPUTE, started_at)
    validate_metrics(step_metrics)
    state = replace(
        state,
        optimizer_step=state.optimizer_step + 1,
        microstep=0,
        phase=TrainingPhase.OPTIMIZER_COMPLETE,
    )
    trainer.set_loop_state(state)
    reporting.begin_step(state.optimizer_step, step_metrics)
    if checkpoint_manager is None:
        raise RuntimeError("checkpoint manager is required at optimizer boundaries")
    complete_step_work(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting,
        lifecycle,
        timer,
        trainer.intervals.log_every_steps,
        trainer.intervals.checkpoint_every_steps,
    )
