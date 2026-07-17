import math
from dataclasses import dataclass, replace
from typing import Protocol

from ..data.prefetch import DataPipelineState
from ..data.records import BeetleBatch
from .callbacks import (
    CancellationRequested,
    ProgressEvent,
    TrainingCallbacks,
    TrainingMetric,
)
from .checkpoint import CheckpointManager, CheckpointPayload
from .state import LoopState, StageKind, TrainingPhase


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
    ) -> CheckpointPayload: ...


def run_continuously(
    pipeline: TrainingPipeline,
    trainer: StageTrainer,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
) -> LoopState:
    try:
        callbacks.check_cancel()
        while True:
            state = trainer.loop_state()
            if state.phase is TrainingPhase.GENERATOR_COMPLETE:
                _complete_accumulation(trainer, pipeline, callbacks, checkpoint_manager)
                continue
            if state.phase in (
                TrainingPhase.OPTIMIZER_COMPLETE,
                TrainingPhase.CHECKPOINTING,
            ):
                _complete_step_work(
                    trainer, pipeline, callbacks, checkpoint_manager
                )
                continue
            _run_batch(pipeline, trainer, callbacks, checkpoint_manager)
    except CancellationRequested:
        state = trainer.loop_state()
        payload = trainer.checkpoint_payload(state, pipeline.state_dict())
        checkpoint_manager.save(payload)
        return state


def _run_batch(
    pipeline: TrainingPipeline,
    trainer: StageTrainer,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
) -> None:
    state = trainer.loop_state()
    resume_discriminator = state.phase in (
        TrainingPhase.DISCRIMINATOR_COMPLETE,
        TrainingPhase.GENERATOR_BACKWARD,
    )
    batch = pipeline.next_batch()
    if not resume_discriminator:
        state = replace(state, phase=TrainingPhase.BATCH_FETCHED)
        _announce(trainer, callbacks, state, ())
    metrics: tuple[TrainingMetric, ...] = ()
    if trainer.trains_discriminator and not resume_discriminator:
        state = replace(state, phase=TrainingPhase.DISCRIMINATOR_BACKWARD)
        _announce(trainer, callbacks, state, ())
        discriminator_metrics = trainer.discriminator_backward(batch)
        _validate_metrics(discriminator_metrics)
        metrics += discriminator_metrics
        state = replace(state, phase=TrainingPhase.DISCRIMINATOR_COMPLETE)
        _announce(trainer, callbacks, state, discriminator_metrics)
    if trainer.loop_state().phase is not TrainingPhase.GENERATOR_BACKWARD:
        state = replace(state, phase=TrainingPhase.GENERATOR_BACKWARD)
        _announce(trainer, callbacks, state, ())
    generator_metrics = trainer.generator_backward(batch)
    _validate_metrics(generator_metrics)
    metrics += generator_metrics
    pipeline.mark_consumed()
    state = _advance_sampler(state, pipeline.state_dict())
    state = replace(
        state,
        microstep=state.microstep + 1,
        phase=TrainingPhase.GENERATOR_COMPLETE,
    )
    _announce(trainer, callbacks, state, generator_metrics)
    _complete_accumulation(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        metrics,
    )


def _complete_accumulation(
    trainer: StageTrainer,
    pipeline: TrainingPipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager | None,
    batch_metrics: tuple[TrainingMetric, ...] = (),
) -> None:
    state = trainer.loop_state()
    if state.microstep < trainer.accumulation_steps:
        ready = replace(state, phase=TrainingPhase.READY)
        trainer.set_loop_state(ready)
        callbacks.check_cancel()
        return
    if state.microstep > trainer.accumulation_steps:
        raise ValueError("microstep exceeds configured accumulation_steps")
    step_metrics = trainer.optimizer_step(state.optimizer_step)
    _validate_metrics(step_metrics)
    metrics = batch_metrics + step_metrics
    state = replace(
        state,
        optimizer_step=state.optimizer_step + 1,
        microstep=0,
        phase=TrainingPhase.OPTIMIZER_COMPLETE,
    )
    report_metrics = (
        metrics
        if _is_due(state.optimizer_step, trainer.intervals.log_every_steps)
        else ()
    )
    _announce(trainer, callbacks, state, report_metrics)
    if checkpoint_manager is None:
        raise RuntimeError("checkpoint manager is required at optimizer boundaries")
    _complete_step_work(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
    )


def _complete_step_work(
    trainer: StageTrainer,
    pipeline: TrainingPipeline,
    callbacks: TrainingCallbacks,
    checkpoint_manager: CheckpointManager,
) -> None:
    state = trainer.loop_state()
    checkpoint_due = _is_due(
        state.optimizer_step, trainer.intervals.checkpoint_every_steps
    )
    if checkpoint_due:
        state = replace(state, phase=TrainingPhase.CHECKPOINTING)
        _announce(trainer, callbacks, state, ())
        complete = replace(state, phase=TrainingPhase.CHECKPOINT_COMPLETE)
        payload = trainer.checkpoint_payload(complete, pipeline.state_dict())
        checkpoint_manager.save(payload)
        _announce(trainer, callbacks, complete, ())
        state = complete
    ready = replace(state, phase=TrainingPhase.READY)
    trainer.set_loop_state(ready)
    callbacks.check_cancel()


def _advance_sampler(state: LoopState, sampler: DataPipelineState) -> LoopState:
    planner = sampler.planner
    return replace(
        state,
        sampler_cursor=planner.batch_index,
        cycle=max(planner.sentence.cycle_index, planner.mid_sentence.cycle_index),
        batch_index=planner.batch_index,
    )


def _announce(
    trainer: StageTrainer,
    callbacks: TrainingCallbacks,
    state: LoopState,
    metrics: tuple[TrainingMetric, ...],
) -> None:
    trainer.set_loop_state(state)
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


def _validate_metrics(metrics: tuple[TrainingMetric, ...]) -> None:
    names = tuple(metric.name for metric in metrics)
    if len(set(names)) != len(names):
        raise ValueError(f"training metric names must be unique: {names}")
    invalid = tuple(
        metric.name for metric in metrics if not math.isfinite(metric.value)
    )
    if invalid:
        raise FloatingPointError(f"non-finite training metrics: {invalid}")


def _is_due(optimizer_step: int, interval: int) -> bool:
    return optimizer_step > 0 and optimizer_step % interval == 0
