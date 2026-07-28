import os
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import pynvml
from mlflow import MlflowClient
from mlflow.entities import Metric

from .config import BeetleConfig
from .logger import logger

BYTES_PER_GIBIBYTE = 1024**3


@dataclass(frozen=True)
class PendingMetrics:
    operation: object
    metrics: tuple[Metric, ...]


class SystemMetricsSampler:
    def __init__(self, device_index: int) -> None:
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.process = psutil.Process()
        psutil.cpu_percent()

    def sample(self) -> dict[str, float]:
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
        return {
            "system/cpu_utilization_percent": psutil.cpu_percent(),
            "system/memory_utilization_percent": system_memory.percent,
            "system/process_rss_gb": process_memory.rss / BYTES_PER_GIBIBYTE,
            "system/gpu_utilization_percent": float(gpu_utilization.gpu),
            "system/gpu_memory_utilization_percent": float(
                gpu_utilization.memory
            ),
            "system/gpu_memory_used_gb": gpu_memory.used / BYTES_PER_GIBIBYTE,
            "system/gpu_temperature_celsius": float(
                pynvml.nvmlDeviceGetTemperature(
                    self.gpu,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            ),
            "system/gpu_power_watts": (
                pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000
            ),
        }


class MlflowLogger:
    def __init__(
        self,
        config: BeetleConfig,
        device_index: int,
        run_id: str | None = None,
    ) -> None:
        self.client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = self.client.get_experiment_by_name("beetle2_training")
        experiment_id = (
            self.client.create_experiment("beetle2_training")
            if experiment is None
            else experiment.experiment_id
        )
        if run_id is None:
            stage = config.training.stage.value
            run = self.client.create_run(
                experiment_id,
                tags={"mlflow.runName": f"beetle2-{stage}"},
            )
            self.run_id = str(run.info.run_id)
            self.client.log_dict(
                self.run_id,
                config.model_dump(mode="json"),
                "config.json",
            )
        else:
            run = self.client.get_run(run_id)
            if run.info.status != "RUNNING":
                self.client.update_run(run_id, status="RUNNING")
            self.run_id = run_id
        self.pending: list[PendingMetrics] = []
        self.retry_metrics: list[tuple[Metric, ...]] = []
        self.retry_artifacts: list[tuple[Path, str]] = []
        self.failure_count = 0
        self.last_error = ""
        self.finished = False
        self.system = SystemMetricsSampler(device_index)

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        timestamp = int(time.time() * 1000)
        metrics = tuple(
            Metric(name, float(value), timestamp, step)
            for name, value in values.items()
        )
        self._submit_metrics(metrics)
        if len(self.pending) >= 128:
            self.flush()

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        try:
            self.client.log_artifact(self.run_id, str(path), artifact_path)
        except Exception as error:
            self.retry_artifacts.append((path, artifact_path))
            self._record_failure(error)

    def flush(self) -> None:
        retry_metrics = self.retry_metrics
        self.retry_metrics = []
        for metrics in retry_metrics:
            self._submit_metrics(metrics)
        pending = self.pending
        self.pending = []
        for item in pending:
            try:
                item.operation.wait()
            except Exception as error:
                self.retry_metrics.append(item.metrics)
                self._record_failure(error)
        retry_artifacts = self.retry_artifacts
        self.retry_artifacts = []
        for path, artifact_path in retry_artifacts:
            self.log_artifact(path, artifact_path)

    def finish(self) -> None:
        self.flush()
        try:
            self.client.set_terminated(self.run_id, "FINISHED")
        except Exception as error:
            self._record_failure(error)
        self.finished = True

    def fail(self) -> None:
        if self.finished:
            return
        self.flush()
        try:
            self.client.set_terminated(self.run_id, "FAILED")
        except Exception as error:
            self._record_failure(error)

    def sample_system_metrics(self) -> dict[str, float]:
        try:
            return self.system.sample()
        except Exception as error:
            self._record_failure(error)
            return {}

    def health_metrics(self) -> dict[str, float]:
        return {
            "recovery/mlflow_failures": float(self.failure_count),
            "overhead/pending_metric_operations": float(
                len(self.pending) + len(self.retry_metrics)
            ),
            "overhead/pending_artifact_jobs": float(len(self.retry_artifacts)),
        }

    def _submit_metrics(self, metrics: tuple[Metric, ...]) -> None:
        try:
            operation = self.client.log_batch(
                self.run_id,
                metrics,
                synchronous=False,
            )
            self.pending.append(PendingMetrics(operation, metrics))
        except Exception as error:
            self.retry_metrics.append(metrics)
            self._record_failure(error)

    def _record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        message = f"{type(error).__name__}: {error}"
        if message != self.last_error:
            logger.warning("MLflow telemetry deferred: %s", message)
            self.last_error = message
