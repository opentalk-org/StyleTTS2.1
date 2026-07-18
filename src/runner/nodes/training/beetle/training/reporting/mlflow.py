import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from mlflow import MlflowClient
from mlflow.entities import Metric

from ..state import StageKind
from .metrics import TrainingMetric

EXPERIMENT_NAME = "beetle_training"
MAX_PENDING_OPERATIONS = 256


class PendingOperation(Protocol):
    def wait(self) -> None: ...


class TrackingClient(Protocol):
    def get_experiment_by_name(self, name: str) -> object | None: ...

    def create_experiment(self, name: str) -> str: ...

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> object: ...

    def get_run(self, run_id: str) -> object: ...

    def log_dict(self, run_id: str, value: object, artifact_file: str) -> object: ...

    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric],
        *,
        synchronous: bool,
    ) -> PendingOperation: ...

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str,
    ) -> object: ...

    def set_terminated(self, run_id: str, status: str) -> object: ...


class MlflowSession:
    def __init__(self, client: TrackingClient, run_id: str) -> None:
        if not run_id:
            raise ValueError("MLflow run ID must not be empty")
        self.client = client
        self.run_id = run_id
        self._pending: list[PendingOperation] = []
        self._finished = False

    @classmethod
    def start(
        cls,
        client: TrackingClient,
        stage: StageKind,
        resolved_config: Mapping[str, Any],
    ) -> "MlflowSession":
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = (
            client.create_experiment(EXPERIMENT_NAME)
            if experiment is None
            else str(experiment.experiment_id)
        )
        tags = {
            "mlflow.runName": f"beetle-{stage.value}",
            "beetle.stage": stage.value,
        }
        run = client.create_run(experiment_id, tags=tags)
        run_id = str(run.info.run_id)
        client.log_dict(run_id, dict(resolved_config), "config.json")
        return cls(client, run_id)

    @classmethod
    def resume(
        cls,
        client: TrackingClient,
        run_id: str,
        stage: StageKind,
    ) -> "MlflowSession":
        run = client.get_run(run_id)
        recorded_stage = run.data.tags["beetle.stage"]
        if recorded_stage != stage.value:
            raise ValueError(
                f"MLflow run stage mismatch: {recorded_stage} != {stage.value}"
            )
        if run.info.status not in ("RUNNING", "FINISHED"):
            raise ValueError(f"MLflow run is not active: {run.info.status}")
        session = cls(client, run_id)
        session._finished = run.info.status == "FINISHED"
        return session

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, metrics: tuple[TrainingMetric, ...], step: int) -> None:
        if self._finished:
            raise ValueError("cannot submit metrics to a finished MLflow run")
        if step <= 0:
            raise ValueError("MLflow metric step must be positive")
        names = tuple(metric.name for metric in metrics)
        if len(set(names)) != len(names):
            raise ValueError(f"MLflow metric names must be unique: {names}")
        timestamp = int(time.time() * 1000)
        batch = [
            Metric(metric.name, metric.value, timestamp, step)
            for metric in metrics
        ]
        operation = self.client.log_batch(
            self.run_id,
            batch,
            synchronous=False,
        )
        self._pending.append(operation)
        if len(self._pending) >= MAX_PENDING_OPERATIONS:
            self.flush()

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        self.client.log_artifact(self.run_id, str(path), artifact_path)

    def flush(self) -> None:
        pending = self._pending
        self._pending = []
        first_error: Exception | None = None
        for operation in pending:
            try:
                operation.wait()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def finish(self) -> None:
        self.flush()
        if not self._finished:
            self.client.set_terminated(self.run_id, "FINISHED")
            self._finished = True

    def fail(self) -> None:
        if self._finished:
            return
        try:
            self.flush()
        finally:
            self.client.set_terminated(self.run_id, "FAILED")


def configured_mlflow_client() -> MlflowClient:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    return MlflowClient(tracking_uri=tracking_uri)
