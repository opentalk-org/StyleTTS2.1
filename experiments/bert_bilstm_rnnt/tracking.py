from __future__ import annotations

import json
import os
import time
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Metric


class ExperimentRun:
    def __init__(self, name: str, config: dict) -> None:
        self.client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = self.client.get_experiment_by_name("bert_bilstm_rnnt")
        experiment_id = self.client.create_experiment("bert_bilstm_rnnt") if experiment is None else experiment.experiment_id
        run = self.client.create_run(experiment_id, tags={"mlflow.runName": name})
        self.run_id = run.info.run_id
        self.client.log_dict(self.run_id, json.loads(json.dumps(config, default=str)), "config.json")

    def metrics(self, values: dict[str, float], step: int) -> None:
        timestamp = int(time.time() * 1000)
        metrics = [Metric(key, float(value), timestamp, step) for key, value in values.items()]
        self.client.log_batch(self.run_id, metrics=metrics)

    def artifact(self, path: Path, destination: str) -> None:
        self.client.log_artifact(self.run_id, str(path), destination)

    def close(self, failed: bool = False) -> None:
        self.client.set_terminated(self.run_id, "FAILED" if failed else "FINISHED")
