"""Tracker protocol and local or GiveMeData-backed adapters."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol

from givemedata_client import MetricsStream

logger = logging.getLogger(__name__)
MetricValue = float | Sequence[float]


class TrackerRun(Protocol):
    """Metric and artifact sink shared by training implementations."""

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None: ...

    def track_metrics(
        self, metrics: Mapping[str, MetricValue], step: int, epoch: int | None = None
    ) -> None: ...

    def log_artifact(self, path: Path, artifact_path: str, step: int) -> None: ...

    def log_artifacts(self, path: Path, artifact_path: str, step: int) -> None: ...

    def close(self) -> None: ...


class GiveMeDataTracker:
    def __init__(self, stream: MetricsStream) -> None:
        self._stream = stream

    def track(
        self,
        value: object,
        name: str,
        step: int,
        epoch: int | None = None,
    ) -> None:
        if isinstance(value, (bool, int, float)):
            self.track_metrics({name: float(value)}, step=step, epoch=epoch)

    def track_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        step: int,
        epoch: int | None = None,
    ) -> None:
        timestamp_unix_ms = time.time_ns() // 1_000_000
        record: dict[str, float | list[float]] = {
            "step": float(step),
            "time": timestamp_unix_ms / 1000.0,
        }
        if epoch is not None:
            record["epoch"] = float(epoch)
        record.update({
            name: float(value) if isinstance(value, (int, float)) else list(value)
            for name, value in metrics.items()
        })
        logger.info("METRICS %s", json.dumps(record, sort_keys=True))
        self._stream.log_metrics(
            metrics,
            step=step,
            timestamp_unix_ms=timestamp_unix_ms,
        )

    def log_artifact(self, path: Path, artifact_path: str, step: int) -> None:
        name = PurePosixPath(artifact_path, Path(path).name).as_posix()
        self._stream.log_artifact(Path(path), name, step)

    def log_artifacts(self, path: Path, artifact_path: str, step: int) -> None:
        self._stream.log_artifacts(Path(path), artifact_path, step)

    def close(self) -> None:
        self._stream.close()


class LocalTracker:
    """File-based tracker: metrics appended to log_dir/metrics.jsonl,
    artifacts copied under log_dir/artifacts/."""

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None:
        if isinstance(value, (bool, int, float)):
            self.track_metrics({name: float(value)}, step=step, epoch=epoch)

    def track_metrics(
        self, metrics: Mapping[str, MetricValue], step: int, epoch: int | None = None
    ) -> None:
        record: dict[str, float | list[float]] = {
            "step": float(step),
            "time": time.time(),
        }
        if epoch is not None:
            record["epoch"] = float(epoch)
        record.update({
            name: float(value) if isinstance(value, (int, float)) else list(value)
            for name, value in metrics.items()
        })
        serialized = json.dumps(record, sort_keys=True)
        self._metrics.write(serialized + "\n")
        self._metrics.flush()
        logger.info("METRICS %s", serialized)

    def log_artifact(self, path: Path, artifact_path: str, step: int) -> None:
        dest = self.log_dir / "artifacts" / artifact_path
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / Path(path).name)

    def log_artifacts(self, path: Path, artifact_path: str, step: int) -> None:
        shutil.copytree(path, self.log_dir / "artifacts" / artifact_path, dirs_exist_ok=True)

    def close(self) -> None:
        self._metrics.close()
