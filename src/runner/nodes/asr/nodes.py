from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.core.types import UnionDataType
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.asr.audio import wav_duration, write_segment_wavs, write_temp_wav
from runner.nodes.asr.parakeet import load_parakeet_model, transcribe_wavs_to_segments
from runner.nodes.asr.whisper import load_whisper_model, transcribe_wav_to_segments, transcribe_wav_to_text
from runner.nodes.datatypes import AUDIO, SEGMENT_GROUP, TRANSCRIPT
from runner.nodes.models import Audio, AudioSegment, SegmentGroup, Transcript, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud


ASR_SOURCE = UnionDataType("ASR_SOURCE", (AUDIO, SEGMENT_GROUP), "Audio or segment group for transcription", "#DC2626")


class TranscribeSettings(StrictSettings):
    language: str = "auto"
    batch_size: int = Field(default=16, ge=1, le=128)
    model_cache_dir: str = Field(default="", title="Model cache directory")
    segment_batch_size: int = Field(default=16, ge=1, le=128)


class TranscribeNode(Node):
    CATEGORY = "Audio / ASR"
    MODEL_NAME = "asr"
    SETTINGS = TranscribeSettings
    INPUTS = {"source": Port("source", ASR_SOURCE)}
    OUTPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator")

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._model: Any | None = None

    async def setup(self, context: Any) -> None:
        cache_dir = self._cache_dir(context)
        self._model = self._load_model(cache_dir)

    async def execute(self, batch, context):
        if self._model is None:
            await self.setup(context)
        outputs = []
        for index, inputs in enumerate(batch, start=1):
            source = inputs["source"]
            await context.report_progress(self.id, index, len(batch), f"{self.MODEL_NAME} transcribed {index}/{len(batch)}")
            outputs.append({"transcript": self._transcribe_source(source, context)})
        return outputs

    def _cache_dir(self, context: Any) -> Path:
        if self.settings.model_cache_dir:
            return Path(self.settings.model_cache_dir)
        return context.cache_dir / "asr" / self.MODEL_NAME

    def _load_model(self, cache_dir: Path) -> Any:
        raise NotImplementedError

    def _transcribe_source(self, source: Audio | SegmentGroup, context: Any) -> Transcript:
        if isinstance(source, Audio):
            return self._transcribe_audio(source)
        if isinstance(source, SegmentGroup):
            return self._transcribe_group(source)
        raise TypeError(f"unsupported ASR source: {type(source).__name__}")

    def _transcribe_audio(self, audio: Audio) -> Transcript:
        path = write_temp_wav(audio.data)
        try:
            spans = self._transcribe_full_path(path, audio.duration)
        finally:
            path.unlink(missing_ok=True)
        return _audio_transcript(self.MODEL_NAME, audio, spans, self.settings.language)

    def _transcribe_group(self, group: SegmentGroup) -> Transcript:
        data = _read_group_audio(group)
        ranges = [(segment.start, segment.end) for segment in group.segments]
        paths = write_segment_wavs(data, ranges)
        try:
            texts = self._transcribe_segment_paths(paths)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
        spans = _group_spans(group.segments, texts)
        return _group_transcript(self.MODEL_NAME, group, spans, self.settings.language)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        raise NotImplementedError

    def _transcribe_segment_paths(self, paths: list[Path]) -> list[str]:
        raise NotImplementedError


class WhisperTranscribeNode(TranscribeNode):
    NODE_TYPE = "WhisperTranscribe"
    MODEL_NAME = "whisper"

    def _load_model(self, cache_dir: Path) -> Any:
        return load_whisper_model(cache_dir)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        return transcribe_wav_to_segments(self._model, path, duration_sec)

    def _transcribe_segment_paths(self, paths: list[Path]) -> list[str]:
        return [transcribe_wav_to_text(self._model, path) for path in paths]


class ParakeetTranscribeNode(TranscribeNode):
    NODE_TYPE = "ParakeetTranscribe"
    MODEL_NAME = "parakeet"

    def _load_model(self, cache_dir: Path) -> Any:
        return load_parakeet_model(cache_dir)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        batches = transcribe_wavs_to_segments(self._model, [path], [duration_sec])
        return batches[0] if batches else []

    def _transcribe_segment_paths(self, paths: list[Path]) -> list[str]:
        durations = [wav_duration(path.read_bytes()) for path in paths]
        batches = transcribe_wavs_to_segments(self._model, paths, durations)
        return [_joined_text(spans) for spans in batches]


def _read_group_audio(group: SegmentGroup) -> bytes:
    source_ids = {segment.source_audio_id for segment in group.segments}
    assert len(source_ids) == 1, f"segment group has multiple source audio ids: {group.id}"
    with database_session() as session:
        return audio_crud.read_audio_file(session, next(iter(source_ids)))


def _audio_transcript(
    model_name: str,
    audio: Audio,
    spans: list[tuple[float, float, str]],
    language: str,
) -> Transcript:
    filtered = [(start, end, text) for start, end, text in spans if text.strip()]
    text = _joined_text(filtered)
    transcript_id = stable_id("transcript", model_name, audio.audio_file_id, audio.id)
    return Transcript(
        text=text,
        model=model_name,
        source_audio_id=audio.audio_file_id,
        start=audio.start,
        end=audio.end,
        speaker=None,
        id=transcript_id,
        lineage_id=stable_id("lineage", model_name, audio.lineage_id),
        segments=[_span_payload(transcript_id, index, start, end, text) for index, (start, end, text) in enumerate(filtered)],
        metadata={**audio.metadata, "language": language, "sample_rate": audio.sample_rate, "channels": audio.channels},
    )


def _group_transcript(
    model_name: str,
    group: SegmentGroup,
    spans: list[tuple[AudioSegment, str]],
    language: str,
) -> Transcript:
    text = " ".join(item_text.strip() for _segment, item_text in spans if item_text.strip()).strip()
    source_id = group.segments[0].source_audio_id
    transcript_id = stable_id("transcript", model_name, group.id)
    return Transcript(
        text=text,
        model=model_name,
        source_audio_id=source_id,
        start=min(segment.start for segment in group.segments),
        end=max(segment.end for segment in group.segments),
        speaker=group.segments[0].speaker,
        id=transcript_id,
        lineage_id=stable_id("lineage", model_name, group.lineage_id),
        segments=[_segment_payload(transcript_id, index, segment, item_text) for index, (segment, item_text) in enumerate(spans)],
        metadata={**group.metadata, "language": language, "source_group_id": group.id},
    )


def _group_spans(segments: list[AudioSegment], texts: list[str]) -> list[tuple[AudioSegment, str]]:
    if len(texts) != len(segments):
        raise ValueError("ASR segment output count mismatch")
    return [(segment, text) for segment, text in zip(segments, texts, strict=True)]


def _span_payload(transcript_id: str, index: int, start: float, end: float, text: str) -> dict[str, Any]:
    return {"id": stable_id("transcript_segment", transcript_id, index), "start": start, "end": end, "text": text, "phon": "", "speaker": ""}


def _segment_payload(transcript_id: str, index: int, segment: AudioSegment, text: str) -> dict[str, Any]:
    return {
        "id": stable_id("transcript_segment", transcript_id, index, segment.segment_id or segment.id),
        "source_segment_id": segment.segment_id or segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": text,
        "phon": segment.phon,
        "speaker": segment.speaker or "",
    }


def _joined_text(spans: list[tuple[float, float, str]]) -> str:
    return " ".join(text.strip() for _start, _end, text in spans if text.strip()).strip()
