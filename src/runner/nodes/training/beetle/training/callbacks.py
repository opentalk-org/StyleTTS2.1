import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Protocol, runtime_checkable

from .state import StageKind, TrainingPhase


class CancellationRequested(RuntimeError):
    """Stop training at the next exact-resume boundary."""


@dataclass(frozen=True)
class TrainingMetric:
    name: str
    value: float


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
