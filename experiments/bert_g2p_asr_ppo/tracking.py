from __future__ import annotations

import json
import os
import time
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Metric


class ExperimentRun:
    def __init__(self, name: str, config: dict, run_id: str | None = None) -> None:
        self.client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = self.client.get_experiment_by_name("bert_g2p_asr_ppo")
        experiment_id = self.client.create_experiment("bert_g2p_asr_ppo") if experiment is None else experiment.experiment_id
        if run_id is None:
            run = self.client.create_run(experiment_id, tags={"mlflow.runName": name})
            self.run_id = run.info.run_id
            self.client.log_dict(self.run_id, _jsonable(config), "config.json")
        else:
            self.run_id = run_id
            self.client.update_run(run_id, status="RUNNING")

    def metrics(self, values: dict[str, float], step: int) -> None:
        timestamp = int(time.time() * 1000)
        self.client.log_batch(self.run_id, metrics=[Metric(key, float(value), timestamp, step) for key, value in values.items()])

    def artifact(self, path: Path, destination: str) -> None:
        self.client.log_artifact(self.run_id, str(path), destination)

    def close(self, failed: bool = False) -> None:
        self.client.set_terminated(self.run_id, "FAILED" if failed else "FINISHED")


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))
