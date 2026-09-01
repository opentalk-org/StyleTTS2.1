from __future__ import annotations

import mimetypes
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import givemedata_pb2 as pb

ARTIFACT_CHUNK_BYTES = 2 * 1024 * 1024
METRICS_QUEUE_SIZE = 16


@dataclass(frozen=True)
class ArtifactFileJob:
    path: Path
    name: str
    step: int
    content_type: str | None
    timestamp_unix_ms: int


@dataclass(frozen=True)
class ArtifactDirectoryJob:
    path: Path
    name: str
    step: int
    timestamp_unix_ms: int


ArtifactJob = ArtifactFileJob | ArtifactDirectoryJob


class MetricsStream:
    def __init__(self, stub, training_id: str) -> None:
        self._requests: queue.Queue[pb.MetricsRequest | None] = queue.Queue(
            maxsize=METRICS_QUEUE_SIZE
        )
        self._requests.put(
            pb.MetricsRequest(
                metadata=pb.MetricsStreamMetadata(training_id=training_id)
            )
        )
        self._future = stub.Metrics.future(self._request_iterator())
        self._artifact_jobs: queue.SimpleQueue[ArtifactJob | None] = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._worker_error: Exception | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._artifact_worker,
            name="givemedata-artifacts",
            daemon=True,
        )
        self._worker.start()

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        self.log_metrics(
            {name: value},
            step=step,
            timestamp_unix_ms=timestamp_unix_ms,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        timestamp = (
            _timestamp_unix_ms()
            if timestamp_unix_ms is None
            else timestamp_unix_ms
        )
        with self._lock:
            self._ensure_accepting()
            for name, value in metrics.items():
                self._enqueue(
                    pb.MetricsRequest(
                        metric=pb.ScalarMetric(
                            step=step,
                            timestamp_unix_ms=timestamp,
                            name=name,
                            value=float(value),
                        )
                    )
                )

    def log_artifact(
        self,
        path: Path,
        name: str,
        step: int,
        content_type: str | None = None,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        artifact_name = _artifact_name(name)
        timestamp = (
            _timestamp_unix_ms()
            if timestamp_unix_ms is None
            else timestamp_unix_ms
        )
        with self._lock:
            self._ensure_accepting()
            self._artifact_jobs.put(
                ArtifactFileJob(
                    path=Path(path),
                    name=artifact_name,
                    step=step,
                    content_type=content_type,
                    timestamp_unix_ms=timestamp,
                )
            )

    def log_artifacts(self, path: Path, name: str, step: int) -> None:
        artifact_name = _artifact_name(name)
        with self._lock:
            self._ensure_accepting()
            self._artifact_jobs.put(
                ArtifactDirectoryJob(
                    path=Path(path),
                    name=artifact_name,
                    step=step,
                    timestamp_unix_ms=_timestamp_unix_ms(),
                )
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._artifact_jobs.put(None)
        self._worker.join()
        self._finish_request_stream()
        rpc_error: Exception | None = None
        try:
            self._future.result()
        except Exception as error:
            rpc_error = error
        self._raise_worker_error()
        if rpc_error is not None:
            raise rpc_error

    def _request_iterator(self) -> Iterator[pb.MetricsRequest]:
        while True:
            request = self._requests.get()
            if request is None:
                return
            yield request

    def _ensure_accepting(self) -> None:
        if self._closed:
            raise RuntimeError("metrics stream is closed")
        self._check_stream()

    def _check_stream(self) -> None:
        self._raise_worker_error()
        if self._future.done():
            self._future.result()
            raise RuntimeError("metrics stream ended unexpectedly")

    def _enqueue(self, request: pb.MetricsRequest) -> None:
        while True:
            self._check_stream()
            try:
                self._requests.put(request, timeout=0.1)
                return
            except queue.Full:
                continue

    def _artifact_worker(self) -> None:
        try:
            while True:
                job = self._artifact_jobs.get()
                if job is None:
                    return
                if isinstance(job, ArtifactFileJob):
                    self._upload_file(job)
                else:
                    self._upload_directory(job)
        except Exception as error:
            with self._lock:
                self._worker_error = error

    def _upload_directory(self, job: ArtifactDirectoryJob) -> None:
        for artifact in sorted(
            item for item in job.path.rglob("*") if item.is_file()
        ):
            relative = artifact.relative_to(job.path).as_posix()
            self._upload_file(
                ArtifactFileJob(
                    path=artifact,
                    name=PurePosixPath(job.name, relative).as_posix(),
                    step=job.step,
                    content_type=None,
                    timestamp_unix_ms=job.timestamp_unix_ms,
                )
            )

    def _upload_file(self, job: ArtifactFileJob) -> None:
        size = job.path.stat().st_size
        content_type = (
            job.content_type
            or mimetypes.guess_type(job.path.name)[0]
            or "application/octet-stream"
        )
        self._enqueue(
            pb.MetricsRequest(
                artifact=pb.ArtifactMetric(
                    step=job.step,
                    timestamp_unix_ms=job.timestamp_unix_ms,
                    name=job.name,
                    content_type=content_type,
                    size_bytes=size,
                )
            )
        )
        with job.path.open("rb") as file:
            while chunk := file.read(ARTIFACT_CHUNK_BYTES):
                self._enqueue(
                    pb.MetricsRequest(
                        artifact_chunk=pb.ArtifactChunk(data=chunk)
                    )
                )

    def _finish_request_stream(self) -> None:
        while True:
            try:
                self._requests.put(None, timeout=0.1)
                return
            except queue.Full:
                if self._future.done():
                    return

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("artifact upload failed") from self._worker_error


def _artifact_name(name: str) -> str:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact name must be a relative path without parent components")
    return path.as_posix()


def _timestamp_unix_ms() -> int:
    return time.time_ns() // 1_000_000
