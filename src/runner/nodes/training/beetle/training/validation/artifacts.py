import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from ..reporting import TrainingMetric
from ..state import StageKind
from .render import artifact_names, render_validation_sample
from .types import ValidationResult, ValidationSampleResult


class ArtifactUploader(Protocol):
    def log_artifact(self, path: Path, artifact_path: str) -> None: ...


class ArtifactQueue:
    def __init__(self, workers: int, capacity: int) -> None:
        if workers <= 0 or capacity <= 0:
            raise ValueError("artifact workers and capacity must be positive")
        self.capacity = capacity
        self.executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="beetle-validation",
        )
        self.futures: list[Future[None]] = []
        self.error: Exception | None = None
        self.closed = False

    @property
    def pending_count(self) -> int:
        return len(self.futures)

    def enqueue(self, job: Callable[[], None]) -> None:
        self._raise_error()
        if self.closed:
            raise RuntimeError("artifact queue is closed")
        if len(self.futures) >= self.capacity:
            self._complete_first()
        self.futures.append(self.executor.submit(job))

    def flush(self) -> None:
        while self.futures:
            self._complete_first()
        self._raise_error()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if not self.closed:
                self.closed = True
                self.executor.shutdown(wait=True)
        self._raise_error()

    def _complete_first(self) -> None:
        future = self.futures.pop(0)
        try:
            future.result()
        except Exception as error:
            if self.error is None:
                self.error = error

    def _raise_error(self) -> None:
        if self.error is not None:
            raise self.error


@dataclass(frozen=True)
class MetricManifest:
    name: str
    value: float


@dataclass(frozen=True)
class SampleManifest:
    position: int
    audio_file_id: str
    seed: int
    losses: tuple[MetricManifest, ...]
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ValidationManifest:
    stage: str
    optimizer_step: int
    audio_file_ids: tuple[str, ...]
    aggregates: tuple[MetricManifest, ...]
    samples: tuple[SampleManifest, ...]


@dataclass(frozen=True)
class _SampleJob:
    sample: ValidationSampleResult
    stage: StageKind
    directory: Path
    sample_rate: int
    uploader: ArtifactUploader
    artifact_path: str

    def __call__(self) -> None:
        paths = render_validation_sample(
            self.sample,
            self.stage,
            self.directory,
            self.sample_rate,
        )
        for path in paths:
            self.uploader.log_artifact(path, self.artifact_path)


@dataclass(frozen=True)
class _UploadJob:
    path: Path
    uploader: ArtifactUploader
    artifact_path: str

    def __call__(self) -> None:
        self.uploader.log_artifact(self.path, self.artifact_path)


class ValidationArtifacts:
    def __init__(
        self,
        output_root: Path,
        sample_rate: int,
        uploader: ArtifactUploader,
        queue: ArtifactQueue,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("validation sample rate must be positive")
        self.output_root = output_root
        self.sample_rate = sample_rate
        self.uploader = uploader
        self.queue = queue

    def publish(self, result: ValidationResult) -> None:
        relative_root = Path(
            "validation",
            result.stage.value,
            f"step_{result.step}",
        )
        local_root = self.output_root / relative_root
        local_root.mkdir(parents=True, exist_ok=True)
        for position, sample in enumerate(result.samples):
            sample_name = f"sample_{position}_{sample.audio_file_id}"
            self.queue.enqueue(
                _SampleJob(
                    sample,
                    result.stage,
                    local_root / sample_name,
                    self.sample_rate,
                    self.uploader,
                    str(relative_root / sample_name),
                )
            )
        self.queue.flush()
        manifest_path = local_root / "metrics.json"
        manifest = _manifest(result)
        manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2),
            encoding="utf-8",
        )
        self.queue.enqueue(
            _UploadJob(manifest_path, self.uploader, str(relative_root))
        )
        self.queue.flush()

    def close(self) -> None:
        self.queue.close()


def _manifest(result: ValidationResult) -> ValidationManifest:
    return ValidationManifest(
        result.stage.value,
        result.step,
        tuple(str(sample.audio_file_id) for sample in result.samples),
        _metrics(result.aggregates),
        tuple(
            SampleManifest(
                position,
                str(sample.audio_file_id),
                sample.seed,
                _metrics(sample.losses),
                artifact_names(result.stage),
            )
            for position, sample in enumerate(result.samples)
        ),
    )


def _metrics(values: tuple[TrainingMetric, ...]) -> tuple[MetricManifest, ...]:
    return tuple(MetricManifest(metric.name, metric.value) for metric in values)
