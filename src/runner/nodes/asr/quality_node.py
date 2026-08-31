from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import torch

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.asr.transcript_quality import (
    annotate_transcript_quality,
    load_checkpoint_aligner,
    score_transcript_batch,
)
from runner.nodes.datatypes import AudioPort, CheckpointRefPort, FloatPort
from runner.nodes.models import Audio, typed_checkpoint


class TranscriptQualityNode(Node):
    NODE_TYPE = "TranscriptQuality"
    DESCRIPTION = "Score phonemized transcripts against their audio with the text aligner embedded in a StyleTTS2 checkpoint. Emits a calibrated error score from 0 (strong agreement) to 1 (strong mismatch) and stores component metrics on the audio metadata."
    CATEGORY = "ASR"
    INPUTS = {"checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST), "audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "score": FloatPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 8},
        keep_loaded=True,
        exclusive_group="accelerator",
    )

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._model = None
        self._cleaner = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_checkpoint_id: UUID | None = None

    async def teardown(self, context: Any) -> None:
        self._model = None
        self._cleaner = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_model(checkpoint)
        audios = [inputs["audio"] for inputs in batch]
        assert all(isinstance(audio, Audio) for audio in audios), "transcript quality inputs must be Audio"
        context.check_cancel()
        metrics = await asyncio.to_thread(
            score_transcript_batch,
            self._model,
            self._cleaner,
            audios,
            self._device,
        )
        return [
            {"audio": annotate_transcript_quality(audio, values), "score": values.score}
            for audio, values in zip(audios, metrics, strict=True)
        ]

    async def _ensure_model(self, checkpoint) -> None:
        if self._model is not None and checkpoint.checkpoint_id == self._loaded_checkpoint_id:
            return
        self._model, self._cleaner = await asyncio.to_thread(load_checkpoint_aligner, checkpoint, self._device)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id
