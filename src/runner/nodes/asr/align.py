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
from runner.nodes.asr.batch import TemporaryAudioBatch
from runner.nodes.asr.confidence import mean_word_confidence
from runner.nodes.asr.whisperx import AlignmentRequest, align_wavs, load_whisperx_align_model, whisperx_device
from runner.nodes.datatypes import AudioPort, CheckpointRefPort
from runner.nodes.models import Audio, AudioSegment, CheckpointRef, typed_checkpoint


class WhisperXAlignSettings(StrictSettings):
    language: str = Field(default="en", title="Language")


class WhisperXAlignNode(Node):
    NODE_TYPE = "WhisperXAlign"
    DESCRIPTION = "Force-align the words of each existing transcript segment against the audio to get precise per-word timings. Takes an alignment checkpoint and audio that already carries text segments, and outputs the same audio with each segment's word-level timing (and confidence) rewritten, replacing any alignment that was there before. Use it after transcription when you need accurate word boundaries. Set the language to match the audio."
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
        audios = [inputs["audio"] for inputs in batch]
        context.check_cancel()
        aligned = await asyncio.to_thread(self._align_batch, audios)
        await context.report_progress(
            self.id,
            len(audios),
            len(audios),
            f"whisperx aligned {len(audios)}/{len(audios)}",
        )
        return [{"audio": audio} for audio in aligned]

    async def _ensure_model(self, checkpoint: CheckpointRef) -> None:
        if self._model is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self._model, self._metadata, self._device = await asyncio.to_thread(self._load_model_logged, checkpoint.path)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id

    def _load_model_logged(self, checkpoint_dir: Path) -> tuple[Any, Any, str]:
        device = whisperx_device()
        self.logger.info("loading whisperx align model (%s) on %s", self.settings.language, device)
        model, metadata = load_whisperx_align_model(checkpoint_dir, self.settings.language, device)
        return model, metadata, device

    def _align_batch(self, audios: list[Audio]) -> list[Audio]:
        spans_by_audio = [
            [
                (
                    max(0.0, segment.start - audio.start),
                    max(0.0, segment.end - audio.start),
                    segment.text,
                )
                for segment in audio.segments
            ]
            for audio in audios
        ]
        with TemporaryAudioBatch(audios) as paths:
            requests = [
                AlignmentRequest(path, spans)
                for path, spans in zip(paths, spans_by_audio, strict=True)
            ]
            words_by_audio = align_wavs(
                self._model,
                self._metadata,
                requests,
                self._device or "cpu",
            )
        assert len(words_by_audio) == len(audios), "whisperx batch output mismatch"
        return [
            self._aligned_audio(audio, spans, words)
            for audio, spans, words in zip(
                audios,
                spans_by_audio,
                words_by_audio,
                strict=True,
            )
        ]

    def _aligned_audio(
        self,
        audio: Audio,
        spans: list[tuple[float, float, str]],
        words: list[dict[str, Any]],
    ) -> Audio:
        segments = [
            self._aligned_segment(segment, words, start, end, audio.start)
            for segment, (start, end, _text) in zip(audio.segments, spans, strict=True)
        ]
        return replace(audio, segments=segments)

    def _aligned_segment(
        self, seg: AudioSegment, words: list[dict[str, Any]], rel_start: float, rel_end: float, offset: float
    ) -> AudioSegment:
        inside = _words_in_span(words, rel_start, rel_end)
        confidence = mean_word_confidence(inside)
        return replace(
            seg,
            alignment=_segment_alignment(inside, offset),
            confidence=confidence if confidence is not None else seg.confidence,
        )


def _words_in_span(words: list[dict[str, Any]], rel_start: float, rel_end: float) -> list[dict[str, Any]]:
    return [word for word in words if rel_start <= (word["start"] + word["end"]) / 2 <= rel_end]


def _segment_alignment(words: list[dict[str, Any]], offset: float) -> list[dict[str, Any]] | None:
    aligned = [
        {"word": word["word"], "start": word["start"] + offset, "end": word["end"] + offset}
        for word in words
    ]
    return aligned or None
