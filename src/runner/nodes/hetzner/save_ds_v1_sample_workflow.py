from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BACKEND_URL = "http://localhost:43800"
WORKFLOW_PATH = Path(__file__).with_name("ds_v1_sample_import_workflow.json")


def main() -> None:
    backend_url = os.environ.get("RUNFLOW_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")
    payload = WORKFLOW_PATH.read_bytes()
    request = urllib.request.Request(
        f"{backend_url}/workflows",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not save workflow to {backend_url}: {error}") from error
    print(json.dumps({"id": body["id"], "name": body["name"]}, indent=2))


if __name__ == "__main__":
    main()
