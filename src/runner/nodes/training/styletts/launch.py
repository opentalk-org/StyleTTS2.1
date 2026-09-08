import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

import yaml

from runflow.runtime.cancellation import check_cancel


def create_run(data_config: dict, train_config: str) -> UUID:
    address = os.environ["TENSORLANE_HTTP_ADDR"].rstrip("/")
    parsed_train_config = yaml.safe_load(train_config)
    request = Request(
        f"{address}/runs",
        data=json.dumps({
            "project_id": "55f68f72-f7b3-522e-8eb6-62253946d8c9",
            "name": parsed_train_config["run_name"],
            "data_config": data_config,
            "train_config": parsed_train_config,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"tensorlane rejected run creation: HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"tensorlane HTTP service is unavailable at {address}: {error.reason}"
        ) from error
    return UUID(payload["run_id"])


def train(
    run_id: UUID,
    process_count: int,
    precision: str,
    output_dir: Path,
) -> None:
    precision = {
        "fp32": "no",
        "fp16": "fp16",
        "bf16": "bf16",
    }[precision]
    command = [sys.executable, "-m", "traintts.main"]
    if process_count > 1:
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(process_count),
            "--num_machines",
            "1",
            "--mixed_precision",
            precision,
            "--dynamo_backend",
            "no",
            "--multi_gpu",
            "-m",
            "traintts.main",
        ]
    environment = os.environ.copy()
    environment["TENSORLANE_RUN_ID"] = str(run_id)
    log_path = output_dir / "training.log"
    with log_path.open(mode="w+") as output:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                check_cancel()
                time.sleep(0.1)
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
        output.seek(0)
        training_log = output.read()
    if process.returncode:
        raise RuntimeError(
            f"traintts exited with code {process.returncode}\n{training_log}"
        )
