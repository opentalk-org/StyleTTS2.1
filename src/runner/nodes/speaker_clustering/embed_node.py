from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import Field
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import FetchConfig

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, SpeakerEmbeddingShardRefPort
from runner.nodes.models import Audio, SpeakerEmbeddingShardRef
from runner.nodes.speaker_clustering.embed_batches import (
    bounded_audio_groups,
    validate_embedding_batch,
)
from runner.nodes.speaker_clustering.ecapa_runtime import (
    ECAPA_EMBEDDING_DIMENSION,
    ECAPARuntime,
    prepare_ecapa_batch,
)
from runner.nodes.speaker_clustering.shards import (
    EmbeddingQuality,
    SpeakerEmbeddingRow,
    write_embedding_shard,
)
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFileCreate
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import EmbeddingRunCreate


ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_MODEL_REVISION = "main"
ECAPA_PREPROCESSING_VERSION = "mono-16khz-l2-v1"


class ECAPASpeakerEmbedSettings(StrictSettings):
    model_source: str = ECAPA_MODEL_SOURCE
    model_revision: str = ECAPA_MODEL_REVISION
    preprocessing_version: str = ECAPA_PREPROCESSING_VERSION
    minimum_duration_seconds: float = Field(default=0.25, gt=0.0, le=10.0)
    maximum_batch_seconds: float = Field(default=600.0, gt=0.0, le=3_600.0)


class ECAPASpeakerEmbedNode(Node):
    NODE_TYPE = "ECAPASpeakerEmbed"
    DESCRIPTION = "Embed bounded speaker-segment batches into durable float16 Parquet shards."
    CATEGORY = "Speaker Clustering"
    SETTINGS = ECAPASpeakerEmbedSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"shard": SpeakerEmbeddingShardRefPort()}
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=128,
        max_size=512,
        sort_by="duration",
    )
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 4},
        keep_loaded=True,
        exclusive_group="accelerator",
    )
    QUEUE_MAX_SIZE = 512

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._runtime: ECAPARuntime | None = None
        self._run_id: UUID | None = None
        self._dataset_id: UUID | None = None
        self._expected_count: int | None = None
        self._processed_count = 0
        self._processed_seconds = 0.0

    async def setup(self, context: Any) -> None:
        device = str(context.device)
        encoder = await asyncio.to_thread(
            EncoderClassifier.from_hparams,
            source=self.settings.model_source,
            fetch_config=FetchConfig(revision=self.settings.model_revision),
            run_opts={"device": device},
        )
        self._runtime = ECAPARuntime(encoder)

    async def teardown(self, context: Any) -> None:
        self._runtime = None
        release_accelerator_memory()

    async def execute(
        self,
        batch: list[dict[str, Any]],
        context: Any,
    ) -> list[dict[str, Any]]:
        if not batch:
            raise ValueError(f"{self.id} requires at least one audio item")
        assert self._runtime is not None, f"{self.id} ECAPA model is not loaded"
        context.check_cancel()
        audios = [inputs["audio"] for inputs in batch]
        run_id = self._ensure_run(audios)
        outputs = []
        groups = bounded_audio_groups(audios, self.settings.maximum_batch_seconds)
        groups.extend(
            [audio]
            for audio in audios
            if audio.duration > self.settings.maximum_batch_seconds
        )
        for group in groups:
            context.check_cancel()
            rows = self._embed_rows(group, context)
            shard_data = write_embedding_shard(rows)
            artifact = self._store_shard(shard_data, run_id, len(rows))
            self._processed_count += len(rows)
            self._processed_seconds += sum(audio.duration for audio in group)
            outputs.append(self._shard_output(run_id, artifact.id, len(rows)))
            await context.report_progress(
                self.id,
                self._processed_count,
                self._expected_count,
                f"embedded {self._processed_count}/{self._expected_count} segments "
                f"({self._processed_seconds:.1f} audio seconds)",
            )
        return outputs

    @property
    def _model_identity(self) -> str:
        return f"{self.settings.model_source}@{self.settings.model_revision}"

    def _ensure_run(self, audios: list[Audio]) -> UUID:
        identity = validate_embedding_batch(audios, self._dataset_id, self._expected_count)
        if self._run_id is not None:
            return self._run_id
        with database_session() as session:
            run = speaker_crud.create_embedding_run(
                session,
                EmbeddingRunCreate(
                    dataset_id=identity.dataset_id,
                    expected_count=identity.source_segment_count,
                    dimension=ECAPA_EMBEDDING_DIMENSION,
                    model_revision=self._model_identity,
                    preprocessing_version=self.settings.preprocessing_version,
                ),
            )
        self._run_id = run.id
        self._dataset_id = identity.dataset_id
        self._expected_count = identity.source_segment_count
        return run.id

    def _embed_rows(self, audios: list[Audio], context: Any) -> list[SpeakerEmbeddingRow]:
        assert self._runtime is not None, f"{self.id} ECAPA model is not loaded"
        accepted = [
            (index, audio)
            for index, audio in enumerate(audios)
            if self.settings.minimum_duration_seconds
            <= audio.duration
            <= self.settings.maximum_batch_seconds
        ]
        rejected = {
            index: "too_short"
            for index, audio in enumerate(audios)
            if audio.duration < self.settings.minimum_duration_seconds
        }
        rejected.update(
            {
                index: "duration_exceeds_maximum_batch_seconds"
                for index, audio in enumerate(audios)
                if audio.duration > self.settings.maximum_batch_seconds
            }
        )
        prepared = None
        if accepted:
            try:
                prepared = prepare_ecapa_batch([audio for _, audio in accepted])
            except (ValueError, RuntimeError):
                accepted, rejected = self._identify_invalid_audio(accepted, rejected, context)
                if accepted:
                    prepared = prepare_ecapa_batch([audio for _, audio in accepted])
        vectors_by_index = {}
        if prepared is not None:
            vectors = self._runtime.embed(prepared)
            vectors_by_index = {
                index: vector for (index, _), vector in zip(accepted, vectors, strict=True)
            }
        return [
            self._row(
                audio,
                vectors_by_index[index] if index in vectors_by_index else None,
                rejected[index] if index in rejected else None,
            )
            for index, audio in enumerate(audios)
        ]

    def _identify_invalid_audio(
        self,
        candidates: list[tuple[int, Audio]],
        rejected: dict[int, str],
        context: Any,
    ) -> tuple[list[tuple[int, Audio]], dict[int, str]]:
        accepted = []
        for index, audio in candidates:
            context.check_cancel()
            try:
                prepare_ecapa_batch([audio])
                accepted.append((index, audio))
            except (ValueError, RuntimeError) as error:
                rejected[index] = _rejection_reason(error)
        return accepted, rejected

    def _row(
        self,
        audio: Audio,
        embedding: np.ndarray | None,
        rejection_reason: str | None,
    ) -> SpeakerEmbeddingRow:
        assert len(audio.segments) == 1, f"{self.id} requires one segment per audio item"
        segment = audio.segments[0]
        segment_id = str(audio.metadata["source_segment_id"])
        true_label = segment.speaker
        return SpeakerEmbeddingRow(
            segment_id=segment_id,
            audio_id=audio.audio_file_id,
            duration_seconds=audio.duration,
            quality=(
                EmbeddingQuality.ACCEPTED
                if rejection_reason is None
                else EmbeddingQuality.REJECTED
            ),
            rejection_reason=rejection_reason,
            true_label=true_label,
            embedding=embedding,
        )

    def _shard_output(self, run_id: UUID, artifact_id: UUID, row_count: int) -> dict[str, Any]:
        return {
            "shard": SpeakerEmbeddingShardRef(
                run_id=run_id,
                artifact_id=artifact_id,
                row_count=row_count,
                dimension=ECAPA_EMBEDDING_DIMENSION,
                model_revision=self._model_identity,
                preprocessing_version=self.settings.preprocessing_version,
            ),
            INPUT_INDEX_OUTPUT: 0,
        }

    def _store_shard(self, data: bytes, run_id: UUID, row_count: int) -> Any:
        with database_session() as session:
            return asset_crud.create_extra_file(
                session,
                ExtraFileCreate(
                    name=f"speaker-embeddings-{run_id}-{self._processed_count}.parquet",
                    data=data,
                    type_="speaker_embedding_shard",
                    metadata={
                        "run_id": str(run_id),
                        "row_count": row_count,
                        "dimension": ECAPA_EMBEDDING_DIMENSION,
                        "model_revision": self._model_identity,
                        "preprocessing_version": self.settings.preprocessing_version,
                    },
                ),
            )


def _rejection_reason(error: Exception) -> str:
    message = str(error)
    if "non-finite" in message:
        return "non_finite"
    if "empty" in message:
        return "empty_audio"
    if "bytes are required" in message:
        return "missing_audio_bytes"
    return "invalid_audio"
