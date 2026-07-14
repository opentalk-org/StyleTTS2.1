from __future__ import annotations

from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import (
    SpeakerEmbeddingSetRefPort,
    SpeakerEmbeddingShardRefPort,
)
from runner.nodes.models import SpeakerEmbeddingSetRef, SpeakerEmbeddingShardRef
from shared.db import database_session
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import EmbeddingShardCreate


class CollectSpeakerEmbeddingsNode(Node):
    NODE_TYPE = "CollectSpeakerEmbeddings"
    DESCRIPTION = "Register embedding shards and emit one reference after the durable run seals."
    CATEGORY = "Speaker Clustering"
    INPUTS = {"shard": SpeakerEmbeddingShardRefPort()}
    OUTPUTS = {"embedding_set": SpeakerEmbeddingSetRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 512

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._run_id: UUID | None = None
        self._emitted = False

    async def execute(
        self,
        batch: list[dict[str, Any]],
        context: Any,
    ) -> list[dict[str, Any]]:
        if not batch:
            raise ValueError(f"{self.id} requires at least one shard")
        shards = [inputs["shard"] for inputs in batch]
        assert all(isinstance(shard, SpeakerEmbeddingShardRef) for shard in shards), (
            f"{self.id} requires SpeakerEmbeddingShardRef input"
        )
        if self._run_id is None:
            self._run_id = shards[0].run_id
        mismatched = [shard.run_id for shard in shards if shard.run_id != self._run_id]
        if mismatched:
            raise ValueError(
                f"{self.id} is bound to embedding run {self._run_id}, "
                f"received {mismatched[0]}"
            )
        outputs = []
        collection = None
        for input_index, shard in enumerate(shards):
            context.check_cancel()
            with database_session() as session:
                collection = speaker_crud.collect_embedding_shard(
                    session,
                    shard.run_id,
                    EmbeddingShardCreate(
                        artifact_id=shard.artifact_id,
                        row_count=shard.row_count,
                        dimension=shard.dimension,
                        model_revision=shard.model_revision,
                        preprocessing_version=shard.preprocessing_version,
                    ),
                )
            if collection.stored_count != collection.expected_count:
                continue
            if self._emitted:
                continue
            self._emitted = True
            outputs.append(
                {
                    "embedding_set": SpeakerEmbeddingSetRef(
                        run_id=collection.run_id,
                        artifact_ids=collection.artifact_ids,
                        dimension=collection.dimension,
                        item_count=collection.stored_count,
                        model_revision=collection.model_revision,
                        preprocessing_version=collection.preprocessing_version,
                    ),
                    INPUT_INDEX_OUTPUT: input_index,
                }
            )
        assert collection is not None, f"{self.id} did not process a shard"
        await context.report_progress(
            self.id,
            collection.stored_count,
            collection.expected_count,
            f"registered {collection.stored_count}/{collection.expected_count} embedding rows",
        )
        return outputs
