import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

from runflow.runtime.cancellation import check_cancel


def create_run(data_config: dict, train_config: str) -> UUID:
    address = os.environ["GIVEMEDATA_HTTP_ADDR"].rstrip("/")
    request = Request(
        f"{address}/trainings",
        data=json.dumps({
            "data_config": data_config,
            "train_config": train_config,
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
            f"givemedata rejected training creation: HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"givemedata HTTP service is unavailable at {address}: {error.reason}"
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
    ]
    if process_count > 1:
        command.append("--multi_gpu")
    command.extend(("-m", "traintts.main"))
    environment = os.environ.copy()
    environment["GIVEMEDATA_RUN_ID"] = str(run_id)
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
