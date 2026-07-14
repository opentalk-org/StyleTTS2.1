from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import (
    SpeakerClusterRunRefPort,
    SpeakerEmbeddingSetRefPort,
)
from runner.nodes.models import SpeakerEmbeddingSetRef
from runner.nodes.speaker_clustering.cluster_runtime.pipeline import (
    ClusterSpeakerEmbeddingsSettings,
    run_clustering_pipeline,
)
from shared.db import database_session
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import EmbeddingRunState


class SpeakerEmbeddingSetSourceSettings(StrictSettings):
    embedding_run_id: UUID


@dataclass(frozen=True)
class ThreadProgressReporter:
    loop: asyncio.AbstractEventLoop
    context: Any
    node_id: str

    def report(self, stage: int, message: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self.context.report_progress(self.node_id, stage, 9, message),
            self.loop,
        )


class SpeakerEmbeddingSetSourceNode(Node):
    NODE_TYPE = "SpeakerEmbeddingSetSource"
    DESCRIPTION = (
        "Load one durable sealed speaker embedding set for clustering or re-clustering."
    )
    CATEGORY = "Speaker Clustering"
    SETTINGS = SpeakerEmbeddingSetSourceSettings
    INPUTS = {}
    OUTPUTS = {"embedding_set": SpeakerEmbeddingSetRefPort()}
    IS_INPUT = True
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=False)

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context: Any) -> int:
        return 0 if self._emitted else 1

    async def execute(
        self, batch: list[dict[str, Any]], context: Any
    ) -> list[dict[str, Any]]:
        if self._emitted:
            return []
        with database_session() as session:
            run = speaker_crud.get_embedding_run(
                session, self.settings.embedding_run_id
            )
            if run.state != EmbeddingRunState.SEALED.value:
                raise ValueError(f"speaker embedding run {run.id} is {run.state}")
            shards = speaker_crud.list_embedding_shards(session, run.id)
            result = SpeakerEmbeddingSetRef(
                run_id=run.id,
                artifact_ids=[shard.artifact_id for shard in shards],
                dimension=run.dimension,
                item_count=run.stored_count,
                model_revision=run.model_revision,
                preprocessing_version=run.preprocessing_version,
            )
        self._emitted = True
        return [{"embedding_set": result}]


class ClusterSpeakerEmbeddingsNode(Node):
    NODE_TYPE = "ClusterSpeakerEmbeddings"
    DESCRIPTION = "Build conservative, exact-reranked speaker clusters from a sealed embedding set."
    CATEGORY = "Speaker Clustering"
    SETTINGS = ClusterSpeakerEmbeddingsSettings
    INPUTS = {"embeddings": SpeakerEmbeddingSetRefPort()}
    OUTPUTS = {"cluster_run": SpeakerClusterRunRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"io": 1, "cpu_workers": 1},
        keep_loaded=False,
    )
    QUEUE_MAX_SIZE = 1

    async def execute(
        self, batch: list[dict[str, Any]], context: Any
    ) -> list[dict[str, Any]]:
        if len(batch) != 1:
            raise ValueError(f"{self.id} requires exactly one sealed embedding set")
        embedding_set = batch[0]["embeddings"]
        assert isinstance(embedding_set, SpeakerEmbeddingSetRef)
        reporter = ThreadProgressReporter(asyncio.get_running_loop(), context, self.id)
        result = await asyncio.to_thread(
            run_clustering_pipeline,
            embedding_set,
            self.settings,
            context.node_dir(self.id),
            context.check_cancel,
            reporter.report,
        )
        return [{"cluster_run": result, INPUT_INDEX_OUTPUT: 0}]
