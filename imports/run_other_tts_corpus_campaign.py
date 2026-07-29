from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from runflow.core.config import RuntimeConfig
from shared.schemas import (
    GraphEdgeRequest,
    GraphNodeRequest,
    InlineGraphRunRequest,
    RunContextRequest,
)

from imports.run_tts_corpus_campaign import (
    DEFAULT_BACKEND,
    DEFAULT_CORPUS,
    _json_request,
    ensure_dataset,
    submit_graph,
    wait_for_run,
)
from runner.nodes.tts.voices import TtsEngine


ENGINES = (
    TtsEngine.CHATTERBOX,
    TtsEngine.F5_TTS,
    TtsEngine.ORPHEUS,
    TtsEngine.DIA,
    TtsEngine.FISH_SPEECH,
    TtsEngine.RAON_OPENTTS,
)
SOURCE_DATASETS = ("tts_piper", "tts_kokoro")


def build_campaign_request(
    engine: TtsEngine,
    corpus_dir: Path,
    source_dataset_ids: tuple[UUID, UUID],
    dataset_id: UUID,
    checkpoint_id: UUID,
    max_jobs: int | None,
    shard_index: int,
    shard_count: int,
    runner_id: str | None,
) -> InlineGraphRunRequest:
    dataset_name = f"tts_{engine.value}"
    nodes = [
        GraphNodeRequest(
            id="synthesis",
            type="OtherTtsCorpusSynthesis",
            params={
                "engine": engine.value,
                "corpus_dir": str(corpus_dir),
                "source_dataset_ids": [
                    str(dataset) for dataset in source_dataset_ids
                ],
                "dataset_id": str(dataset_id),
                "dataset_name": dataset_name,
                "checkpoint_id": str(checkpoint_id),
                "batch_size": 8 if engine is TtsEngine.ORPHEUS else 4,
                "max_jobs": max_jobs,
                "shard_index": shard_index,
                "shard_count": shard_count,
            },
        ),
        GraphNodeRequest(
            id="save",
            type="SaveAudioRecord",
            params={
                "storage_mode": "stored",
                "virtual": False,
                "bulk_import_packs": True,
                "dataset_id": str(dataset_id),
            },
            runtime={
                "queue_max_size": 64,
                "batch_policy": {
                    "mode": "micro_batch",
                    "preferred_size": 64,
                    "max_size": 64,
                    "timeout_ms": 25,
                    "sort_by": None,
                },
            },
        ),
    ]
    edges = [
        GraphEdgeRequest(
            source_node="synthesis",
            source_port="audio",
            target_node="save",
            target_port="audio",
        )
    ]
    context = RunContextRequest(
        config=RuntimeConfig(
            resources={"io": 4, "accelerator": 1, "vram_gb": 30},
        )
    )
    return InlineGraphRunRequest(
        runner_id=runner_id,
        nodes=nodes,
        edges=edges,
        context=context,
    )


def ensure_checkpoint(backend_url: str, engine: TtsEngine) -> UUID:
    matches = [
        checkpoint
        for checkpoint in _json_request(
            backend_url,
            "GET",
            "/checkpoints",
        )
        if checkpoint["type_"] == engine.value
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple {engine.value} checkpoints found"
        )
    if matches:
        return UUID(matches[0]["id"])
    request = InlineGraphRunRequest(
        nodes=[
            GraphNodeRequest(
                id="download",
                type="CatalogDownload",
                params={
                    "catalog_key": "tts_models",
                    "item": engine.value,
                },
            )
        ]
    )
    status = submit_graph(backend_url, request)
    terminal = wait_for_run(backend_url, status["run_id"])
    if terminal["state"] != "succeeded":
        raise RuntimeError(
            f"{engine.value} checkpoint download failed: "
            f"{terminal['error']}"
        )
    matches = [
        checkpoint
        for checkpoint in _json_request(
            backend_url,
            "GET",
            "/checkpoints",
        )
        if checkpoint["type_"] == engine.value
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{engine.value} checkpoint download produced "
            f"{len(matches)} checkpoints"
        )
    return UUID(matches[0]["id"])


def source_dataset_ids(backend_url: str) -> tuple[UUID, UUID]:
    datasets = {
        dataset["name"]: dataset
        for dataset in _json_request(backend_url, "GET", "/datasets")
    }
    missing = sorted(set(SOURCE_DATASETS) - set(datasets))
    if missing:
        raise RuntimeError(
            f"missing TTS source datasets: {','.join(missing)}"
        )
    return tuple(
        UUID(datasets[name]["id"])
        for name in SOURCE_DATASETS
    )


def run_engine(
    backend_url: str,
    corpus_dir: Path,
    sources: tuple[UUID, UUID],
    engine: TtsEngine,
    max_jobs: int | None,
    shard_index: int,
    shard_count: int,
    runner_id: str | None,
) -> dict[str, Any]:
    checkpoint_id = ensure_checkpoint(backend_url, engine)
    dataset_id = ensure_dataset(
        backend_url,
        f"tts_{engine.value}",
    )
    request = build_campaign_request(
        engine,
        corpus_dir,
        sources,
        dataset_id,
        checkpoint_id,
        max_jobs,
        shard_index,
        shard_count,
        runner_id,
    )
    submitted = submit_graph(backend_url, request)
    return wait_for_run(backend_url, submitted["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch resumable FineWiki jobs for the remaining TTS engines"
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
    parser.add_argument(
        "--engine",
        action="append",
        choices=[engine.value for engine in ENGINES],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--runner-id")
    arguments = parser.parse_args()
    if arguments.smoke and arguments.max_jobs is not None:
        parser.error("--smoke and --max-jobs are mutually exclusive")
    selected = (
        tuple(TtsEngine(value) for value in arguments.engine)
        if arguments.engine
        else ENGINES
    )
    max_jobs = 1 if arguments.smoke else arguments.max_jobs
    sources = source_dataset_ids(arguments.backend_url)
    for engine in selected:
        terminal = run_engine(
            arguments.backend_url,
            arguments.corpus_dir,
            sources,
            engine,
            max_jobs,
            arguments.shard_index,
            arguments.shard_count,
            arguments.runner_id,
        )
        if terminal["state"] != "succeeded":
            raise RuntimeError(
                f"{engine.value} campaign failed: {terminal['error']}"
            )


if __name__ == "__main__":
    main()
