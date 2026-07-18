import math
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Protocol, runtime_checkable

from .reporting import ForegroundCategory, StepTimer, TrainingMetric
from .state import LoopState, StageKind, TrainingPhase


class CancellationRequested(RuntimeError):
    """Stop training at the next exact-resume boundary."""


@dataclass(frozen=True)
class ProgressEvent:
    stage: StageKind
    optimizer_step: int
    microstep: int
    phase: TrainingPhase
    metrics: tuple[TrainingMetric, ...]


@dataclass(frozen=True)
class ArtifactEvent:
    path: Path
    media_type: str


@runtime_checkable
class TrainingCallbacks(Protocol):
    def check_cancel(self) -> None: ...

    def report_progress(self, event: ProgressEvent) -> None: ...

    def publish_artifact(self, path: Path, media_type: str) -> None: ...


class StandaloneCallbacks:
    def __init__(
        self,
        progress_sink: Callable[[ProgressEvent], None],
        artifact_sink: Callable[[ArtifactEvent], None],
    ) -> None:
        self.progress_sink = progress_sink
        self.artifact_sink = artifact_sink
        self._cancel = threading.Event()

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def request_cancel(self) -> None:
        self._cancel.set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise CancellationRequested("training cancellation requested")

    def report_progress(self, event: ProgressEvent) -> None:
        self.progress_sink(event)

    def publish_artifact(self, path: Path, media_type: str) -> None:
        self.artifact_sink(ArtifactEvent(path, media_type))

    def _handle_signal(self, signal_number: int, frame: FrameType | None) -> None:
        del signal_number, frame
        self.request_cancel()


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


def report_only(
    callbacks: TrainingCallbacks,
    state: LoopState,
    metrics: tuple[TrainingMetric, ...],
    timer: StepTimer,
) -> None:
    started_at = time.monotonic()
    try:
        callbacks.report_progress(
            ProgressEvent(
                state.stage,
                state.optimizer_step,
                state.microstep,
                state.phase,
                metrics,
            )
        )
    finally:
        timer.record(ForegroundCategory.REPORTING, started_at)
