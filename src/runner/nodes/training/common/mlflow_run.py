from __future__ import annotations

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import time
from typing import Protocol

from mlflow import MlflowClient
from mlflow.entities import Metric

logger = logging.getLogger(__name__)
MAX_PENDING_OPERATIONS = 256


class TrackerRun(Protocol):
    """Metric and artifact sink shared by training implementations."""

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None: ...

    def track_metrics(
        self, metrics: Mapping[str, float], step: int, epoch: int | None = None
    ) -> None: ...

    def log_artifact(self, path: Path, artifact_path: str) -> None: ...

    def close(self) -> None: ...


class TrackingClient(Protocol):
    def log_metric(
        self, run_id: str, key: str, value: float, *, step: int, synchronous: bool
    ) -> "PendingOperation": ...

    def log_batch(
        self, run_id: str, metrics: list[Metric], *, synchronous: bool
    ) -> "PendingOperation": ...

    def log_text(self, run_id: str, text: str, artifact_file: str) -> object: ...

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> object: ...

    def set_terminated(self, run_id: str, status: str) -> object: ...


class PendingOperation(Protocol):
    def wait(self) -> None: ...


class NoopTrackerRun:
    """Keeps training independent from temporary tracking-server outages."""

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None:
        return None

    def track_metrics(
        self, metrics: Mapping[str, float], step: int, epoch: int | None = None
    ) -> None:
        return None

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        return None

    def close(self) -> None:
        return None


class MlflowRun:
    """Concurrency-safe adapter that always addresses one explicit MLflow run."""

    def __init__(self, client: TrackingClient, run_id: str) -> None:
        self._client = client
        self._run_id = run_id
        self._last_epoch_step: tuple[int, int] | None = None
        self._last_logged_step: int | None = None
        self._pending: list[PendingOperation] = []

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None:
        if step != self._last_logged_step:
            operation = self._client.log_metric(
                self._run_id, "step", float(step), step=step, synchronous=False
            )
            self._queue(operation)
            self._last_logged_step = step
        epoch_step = None if epoch is None else (epoch, step)
        if epoch_step is not None and epoch_step != self._last_epoch_step:
            operation = self._client.log_metric(
                self._run_id, "epoch", float(epoch), step=step, synchronous=False
            )
            self._queue(operation)
            self._last_epoch_step = epoch_step
        if isinstance(value, (bool, int, float)):
            operation = self._client.log_metric(
                self._run_id, name, float(value), step=step, synchronous=False
            )
            self._queue(operation)
            return
        artifact_file = f"text/{name}/step_{step:09d}.txt"
        self._client.log_text(self._run_id, str(value), artifact_file)

    def track_metrics(
        self, metrics: Mapping[str, float], step: int, epoch: int | None = None
    ) -> None:
        timestamp = int(time.time() * 1000)
        batch = [Metric(name, float(value), timestamp, step) for name, value in metrics.items()]
        if step != self._last_logged_step:
            batch.append(Metric("step", float(step), timestamp, step))
            self._last_logged_step = step
        epoch_step = None if epoch is None else (epoch, step)
        if epoch_step is not None and epoch_step != self._last_epoch_step:
            batch.append(Metric("epoch", float(epoch), timestamp, step))
            self._last_epoch_step = epoch_step
        operation = self._client.log_batch(self._run_id, batch, synchronous=False)
        self._queue(operation)

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        self._client.log_artifact(self._run_id, str(path), artifact_path)

    def close(self) -> None:
        self._flush_pending()
        self._client.set_terminated(self._run_id, "FINISHED")

    def _queue(self, operation: PendingOperation) -> None:
        self._pending.append(operation)
        if len(self._pending) >= MAX_PENDING_OPERATIONS:
            self._flush_pending()

    def _flush_pending(self) -> None:
        for operation in self._pending:
            operation.wait()
        self._pending.clear()


def start_mlflow_run(*, experiment: str, name: str, config: dict[str, object]) -> TrackerRun:
    """Create an MLflow run without relying on process-global active-run state."""
    try:
        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment_record = client.get_experiment_by_name(experiment)
        experiment_id = (
            client.create_experiment(experiment)
            if experiment_record is None
            else experiment_record.experiment_id
        )
        run = client.create_run(experiment_id, tags={"mlflow.runName": name})
        client.log_dict(run.info.run_id, config, "config.json")
        logger.info("MLflow run started experiment=%s name=%s", experiment, name)
        return MlflowRun(client, run.info.run_id)
    except Exception:
        logger.warning("failed to start MLflow run; training metrics will not be logged", exc_info=True)
        return NoopTrackerRun()
