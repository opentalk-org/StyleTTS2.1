from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import UUID

from runflow.core.config import RuntimeConfig
from shared.schemas import (
    GraphEdgeRequest,
    GraphNodeRequest,
    InlineGraphRunRequest,
    RunContextRequest,
)


DEFAULT_BACKEND = "http://127.0.0.1:8001"
DEFAULT_CORPUS = Path(__file__).parent / "tts_text_data" / "output"
TERMINAL_STATES = {"succeeded", "failed", "stopped"}


def build_campaign_request(
    corpus_dir: Path,
    piper_dataset_id: UUID,
    kokoro_dataset_id: UUID,
    kokoro_checkpoint_id: UUID,
    max_jobs: int | None,
) -> InlineGraphRunRequest:
    common = {
        "corpus_dir": str(corpus_dir),
        "max_jobs": max_jobs,
    }
    nodes = [
        GraphNodeRequest(
            id="piper",
            type="PiperCorpusSynthesis",
            params={
                **common,
                "dataset_id": str(piper_dataset_id),
                "dataset_name": "tts_piper",
                "workers": 15,
                "jobs_per_worker": 8,
            },
        ),
        GraphNodeRequest(
            id="save_piper",
            type="SaveAudioRecord",
            params={
                "storage_mode": "stored",
                "virtual": False,
                "bulk_import_packs": True,
                "dataset_id": str(piper_dataset_id),
            },
            runtime=_save_runtime(),
        ),
        GraphNodeRequest(
            id="kokoro",
            type="KokoroCorpusSynthesis",
            params={
                **common,
                "dataset_id": str(kokoro_dataset_id),
                "dataset_name": "tts_kokoro",
                "checkpoint_id": str(kokoro_checkpoint_id),
                "batch_size": 16,
            },
        ),
        GraphNodeRequest(
            id="save_kokoro",
            type="SaveAudioRecord",
            params={
                "storage_mode": "stored",
                "virtual": False,
                "bulk_import_packs": True,
                "dataset_id": str(kokoro_dataset_id),
            },
            runtime=_save_runtime(),
        ),
    ]
    edges = [
        GraphEdgeRequest(
            source_node="piper",
            source_port="audio",
            target_node="save_piper",
            target_port="audio",
        ),
        GraphEdgeRequest(
            source_node="kokoro",
            source_port="audio",
            target_node="save_kokoro",
            target_port="audio",
        ),
    ]
    context = RunContextRequest(
        config=RuntimeConfig(
            resources={
                "io": 4,
                "cpu_workers": 15,
                "accelerator": 1,
                "vram_gb": 30,
            },
        )
    )
    return InlineGraphRunRequest(nodes=nodes, edges=edges, context=context)


def ensure_dataset(backend_url: str, name: str) -> UUID:
    matches = [
        item
        for item in _json_request(backend_url, "GET", "/datasets")
        if item["name"] == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple datasets named {name!r}")
    if matches:
        return UUID(matches[0]["id"])
    created = _json_request(
        backend_url,
        "POST",
        "/datasets",
        {"name": name},
    )
    return UUID(created["id"])


def ensure_kokoro_checkpoint(backend_url: str) -> UUID:
    checkpoint = _find_kokoro_checkpoint(backend_url)
    if checkpoint is not None:
        return checkpoint
    request = InlineGraphRunRequest(
        nodes=[
            GraphNodeRequest(
                id="download_kokoro",
                type="CatalogDownload",
                params={"catalog_key": "tts_models", "item": "kokoro"},
            )
        ]
    )
    status = submit_graph(backend_url, request)
    terminal = wait_for_run(backend_url, status["run_id"])
    if terminal["state"] != "succeeded":
        raise RuntimeError(
            f"Kokoro checkpoint graph failed: {terminal['error']}"
        )
    checkpoint = _find_kokoro_checkpoint(backend_url)
    if checkpoint is None:
        raise RuntimeError("Kokoro checkpoint graph produced no checkpoint")
    return checkpoint


def submit_graph(
    backend_url: str,
    request: InlineGraphRunRequest,
) -> dict[str, Any]:
    return _json_request(
        backend_url,
        "POST",
        "/graphs/runs",
        request.model_dump(mode="json"),
    )


def wait_for_run(
    backend_url: str,
    run_id: str,
) -> dict[str, Any]:
    while True:
        status = _json_request(
            backend_url,
            "GET",
            f"/runs/{run_id}",
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": status["state"],
                    "events": status["event_count"],
                }
            ),
            flush=True,
        )
        if status["state"] in TERMINAL_STATES:
            return status
        time.sleep(10)


def _find_kokoro_checkpoint(backend_url: str) -> UUID | None:
    matches = [
        checkpoint
        for checkpoint in _json_request(
            backend_url,
            "GET",
            "/checkpoints",
        )
        if checkpoint["type_"] == "kokoro"
    ]
    if len(matches) > 1:
        raise RuntimeError("multiple Kokoro checkpoints found")
    return UUID(matches[0]["id"]) if matches else None


def _save_runtime() -> dict[str, Any]:
    return {
        "queue_max_size": 256,
        "batch_policy": {
            "mode": "micro_batch",
            "preferred_size": 256,
            "max_size": 256,
            "timeout_ms": 25,
            "sort_by": None,
        },
    }


def _json_request(
    backend_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        f"{backend_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} returned {error.code}: {detail}"
        ) from error
    return json.loads(body) if body else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the resumable FineWiki TTS corpus graph"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("RUNFLOW_BACKEND_URL", DEFAULT_BACKEND),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS,
    )
    parser.add_argument("--benchmark-lines", type=int)
    parser.add_argument("--no-wait", action="store_true")
    arguments = parser.parse_args()
    piper_dataset = ensure_dataset(arguments.backend_url, "tts_piper")
    kokoro_dataset = ensure_dataset(arguments.backend_url, "tts_kokoro")
    kokoro_checkpoint = ensure_kokoro_checkpoint(arguments.backend_url)
    graph = build_campaign_request(
        arguments.corpus_dir,
        piper_dataset,
        kokoro_dataset,
        kokoro_checkpoint,
        arguments.benchmark_lines,
    )
    status = submit_graph(arguments.backend_url, graph)
    print(json.dumps(status), flush=True)
    if not arguments.no_wait:
        terminal = wait_for_run(arguments.backend_url, status["run_id"])
        if terminal["state"] != "succeeded":
            raise RuntimeError(
                f"campaign graph failed: {terminal['error']}"
            )


if __name__ == "__main__":
    main()
