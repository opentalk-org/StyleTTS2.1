from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from runner.nodes.training.common.mlflow_run import TrackerRun


class ArtifactUploadQueue:
    """Upload validation media concurrently without blocking GPU evaluation."""

    def __init__(self, run: TrackerRun, workers: int) -> None:
        self._run = run
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="validation-artifact",
        )
        self._pending: list[Future[None]] = []

    def enqueue(self, path: Path, artifact_path: str) -> None:
        future = self._executor.submit(self._run.log_artifact, path, artifact_path)
        self._pending.append(future)

    def flush(self) -> None:
        pending = self._pending
        self._pending = []
        for future in pending:
            future.result()

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)
