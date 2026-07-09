from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.asr.whisperx import align_words, load_whisperx_align_model, whisperx_device
from runner.nodes.datatypes import AudioPort, CheckpointRefPort
from runner.nodes.models import Audio, AudioSegment, CheckpointRef, typed_checkpoint
from shared.log_streams import route_output_to_logger


class WhisperXAlignSettings(StrictSettings):
    language: str = Field(default="en", title="Language")


class WhisperXAlignNode(Node):
    """Force-align the words of each existing segment with WhisperX.

    Consumes audio that already carries segments (text + timing) and rewrites each
    segment's per-word ``alignment`` from the WhisperX aligner, replacing any
    alignment that was there before. The checkpoint is a downloaded wav2vec2 model.
    """

    NODE_TYPE = "WhisperXAlign"
    CATEGORY = "ASR"
    MODEL_NAME = "whisperx"
    SETTINGS = WhisperXAlignSettings
    INPUTS = {"checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST), "audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator")

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._model: Any | None = None
        self._metadata: Any | None = None
        self._loaded_checkpoint_id: UUID | None = None
        self._device: str | None = None

    async def teardown(self, context: Any) -> None:
        self._model = None
        self._metadata = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_model(checkpoint)
        outputs = []
        for index, inputs in enumerate(batch, start=1):
            audio: Audio = inputs["audio"]
            await context.report_progress(self.id, index, len(batch), f"whisperx aligned {index}/{len(batch)}")
            with route_output_to_logger(self.logger):
                aligned = self._align_audio(audio)
            outputs.append({"audio": aligned})
        return outputs

    async def _ensure_model(self, checkpoint: CheckpointRef) -> None:
        if self._model is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self._model, self._metadata, self._device = await asyncio.to_thread(self._load_model_logged, checkpoint.path)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id

    def _load_model_logged(self, checkpoint_dir: Path) -> tuple[Any, Any, str]:
        device = whisperx_device()
        self.logger.info("loading whisperx align model (%s) on %s", self.settings.language, device)
        with route_output_to_logger(self.logger):
            model, metadata = load_whisperx_align_model(checkpoint_dir, self.settings.language, device)
        return model, metadata, device

    def _align_audio(self, audio: Audio) -> Audio:
        if not audio.segments:
            return audio
        assert audio.data is not None, f"audio bytes are required for alignment: {audio.id}"
        spans = [(max(0.0, seg.start - audio.start), max(0.0, seg.end - audio.start), seg.text) for seg in audio.segments]
        path = write_temp_wav(audio.data)
        try:
            words = align_words(self._model, self._metadata, path, spans, self._device or "cpu")
        finally:
            path.unlink(missing_ok=True)
        segments = [
            replace(seg, alignment=_segment_alignment(words, rel_start, rel_end, audio.start))
            for seg, (rel_start, rel_end, _text) in zip(audio.segments, spans, strict=True)
        ]
        return replace(audio, segments=segments)


def _segment_alignment(words: list[dict[str, Any]], rel_start: float, rel_end: float, offset: float) -> list[dict[str, Any]] | None:
    aligned = [
        {"word": word["word"], "start": word["start"] + offset, "end": word["end"] + offset, "score": word.get("score")}
        for word in words
        if rel_start <= (word["start"] + word["end"]) / 2 <= rel_end
    ]
    return aligned or None
