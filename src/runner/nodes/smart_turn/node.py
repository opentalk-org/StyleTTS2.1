from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, BoolPort, CheckpointRefPort, FloatPort
from runner.nodes.models import Audio, CheckpointRef, typed_checkpoint
from runner.nodes.smart_turn.audio import load_audio_bytes, prepare_waveforms
from runner.nodes.smart_turn.inference import (
    SmartTurnBundle,
    is_turn_complete,
    load_smart_turn_bundle,
    predict_probabilities,
)


class SmartTurnPredictNode(Node):
    NODE_TYPE = "SmartTurnPredict"
    DESCRIPTION = "Classify whether each incoming audio clip completes a conversational turn using Smart Turn v3.2. The node preserves every audio item and emits a boolean decision plus completion probability."
    CATEGORY = "Audio"
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "audio": AudioPort(),
    }
    OUTPUTS = {
        "audio": AudioPort(),
        "turn_complete": BoolPort(),
        "probability": FloatPort(),
    }
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=32,
        max_size=64,
        sort_by="duration",
    )
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 128

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._bundle: SmartTurnBundle | None = None
        self._loaded_checkpoint_id: UUID | None = None

    async def teardown(self, context: Any) -> None:
        self._bundle = None
        self._loaded_checkpoint_id = None

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Any]]:
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_bundle(checkpoint)
        audios = [inputs["audio"] for inputs in batch]
        assert all(isinstance(audio, Audio) for audio in audios), "SmartTurnPredict inputs must be Audio"
        context.check_cancel()
        payloads = await asyncio.to_thread(load_audio_bytes, audios)
        waveforms = []
        for start in range(0, len(audios), 8):
            context.check_cancel()
            end = min(start + 8, len(audios))
            waveforms.extend(
                await asyncio.to_thread(
                    prepare_waveforms,
                    audios[start:end],
                    payloads[start:end],
                )
            )
            await context.report_progress(
                self.id,
                end,
                len(audios),
                f"Prepared {end}/{len(audios)} Smart Turn inputs",
            )
        context.check_cancel()
        assert self._bundle is not None, "Smart Turn model bundle is not loaded"
        probabilities = await asyncio.to_thread(predict_probabilities, self._bundle, waveforms)
        return [
            {
                "audio": audio,
                "turn_complete": is_turn_complete(probability),
                "probability": probability,
            }
            for audio, probability in zip(audios, probabilities, strict=True)
        ]

    async def _ensure_bundle(self, checkpoint: CheckpointRef) -> None:
        if checkpoint.metadata["type"] != "smart_turn":
            raise ValueError(f"SmartTurnPredict requires smart_turn checkpoint: {checkpoint.checkpoint_id}")
        if self._bundle is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self.logger.info("loading Smart Turn model checkpoint=%s", checkpoint.path)
        self._bundle = await asyncio.to_thread(load_smart_turn_bundle, checkpoint.path)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id
