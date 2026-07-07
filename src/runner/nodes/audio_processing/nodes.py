from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.asr.audio import extract_wav_range, wav_info
from runner.nodes.assets.model_downloads import single_checkpoint_file
from runner.nodes.datatypes import AUDIO, CHECKPOINT_REF, JSON
from runner.nodes.models import Audio, AudioSegment, CheckpointRef, stable_id
from shared.log_streams import route_output_to_logger
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode, AudioFeatureSettings, analyze_audio_features


class VadSettings(StrictSettings):
    min_segment_sec: float = Field(default=1.0, ge=0.1, le=600.0)
    max_segment_sec: float = Field(default=12.0, ge=1.0, le=600.0)
    padding_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    max_silence_gap_ms: int = Field(default=400, ge=50, le=30000)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    hop_length: int = Field(default=512, ge=64, le=4096)


class CutAudioSettings(StrictSettings):
    fade_ms: int = Field(default=0, ge=0, le=100)


class SortformerSettings(StrictSettings):
    batch_size: int = Field(default=4, ge=1, le=64)
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    min_segment_sec: float = Field(default=0.25, ge=0.0, le=30.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    chunk_len: int | None = Field(default=None, ge=1)
    chunk_right_context: int | None = Field(default=None, ge=0)
    fifo_len: int | None = Field(default=None, ge=1)
    spkcache_update_period: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


class VadDetectNode(Node):
    NODE_TYPE = "VadDetect"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = VadSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            outputs.append({"audio": vad_segments(audio, self.settings)})
        return outputs


class SortformerDiarizationNode(Node):
    NODE_TYPE = "SortformerDiarization"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = SortformerSettings
    INPUTS = {"audio": Port("audio", AUDIO), "checkpoint": Port("checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST)}
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=4, max_size=16)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator")

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._model: Any | None = None
        self._loaded_checkpoint_id = None

    async def teardown(self, context: Any) -> None:
        self._model = None
        self._loaded_checkpoint_id = None

    async def execute(self, batch, context):
        checkpoint = _typed_checkpoint(batch[0]["checkpoint"])
        if self._model is None or self._loaded_checkpoint_id != checkpoint.checkpoint_id:
            self._model = await asyncio.to_thread(self._load_model, checkpoint.path)
            self._loaded_checkpoint_id = checkpoint.checkpoint_id
        audios = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            audios.append(audio)
        with route_output_to_logger(self.logger):
            diarized = diarize_audio_batch(self._model, audios, self.settings)
        return [
            {"audio": speaker_audio_segments(audio, segments, self.settings)}
            for audio, segments in zip(audios, diarized, strict=True)
        ]

    def _load_model(self, checkpoint_dir: Path):
        self.logger.info("loading sortformer model from checkpoint")
        with route_output_to_logger(self.logger):
            return load_sortformer_model(checkpoint_dir, self.settings)


def _typed_checkpoint(value: CheckpointRef | dict[str, Any]) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("Sortformer diarization requires a resolved CheckpointRef")


class CutAudioBySegmentsNode(Node):
    NODE_TYPE = "CutAudioBySegments"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = CutAudioSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            outputs.extend({"audio": item} for item in cut_audio_by_segments(audio, self.settings))
        return outputs


class CalculateAudioStatsNode(Node):
    NODE_TYPE = "CalculateAudioStats"
    CATEGORY = "Audio / Statistics"
    SETTINGS = AudioFeatureSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"stats": Port("stats", JSON)}
    BATCH_POLICY = AnalyzeAudioFeaturesNode.BATCH_POLICY

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            outputs.append({"stats": analyze_audio_features(audio, self.settings.silence_threshold_db, self.settings.hop_length)})
        return outputs


def vad_segments(audio: Audio, settings: VadSettings) -> list[Audio]:
    librosa, np = _audio_processing_dependencies()
    y, sr = librosa.load(BytesIO(audio.data), sr=None, mono=True)
    samples = np.asarray(y, dtype=np.float32)
    if samples.size == 0:
        return []
    intervals = librosa.effects.split(
        samples,
        top_db=abs(float(settings.silence_threshold_db)),
        frame_length=2048,
        hop_length=settings.hop_length,
    )
    spans = _split_long_spans(
        _merge_spans(
            [
                (
                    max(0.0, float(start) / float(sr) - settings.padding_sec),
                    min(audio.duration, float(end) / float(sr) + settings.padding_sec),
                )
                for start, end in intervals
            ],
            settings.max_silence_gap_ms / 1000.0,
        ),
        settings.max_segment_sec,
    )
    valid_spans = [(start, end) for start, end in spans if end - start >= settings.min_segment_sec]
    wav = wav_info(audio.data)
    return [_segment_audio(audio, start, end, wav, index) for index, (start, end) in enumerate(valid_spans)]


def load_sortformer_model(checkpoint_dir: Path, settings: SortformerSettings) -> Any:
    try:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        raise RuntimeError("sortformer_dependencies_not_installed") from exc

    weights = single_checkpoint_file(checkpoint_dir, (".nemo",))
    device = _sortformer_device(torch, settings.device)
    model = SortformerEncLabelModel.restore_from(restore_path=str(weights), map_location=device)
    model = model.to(device)
    model.eval()
    _configure_sortformer_modules(model, settings)
    return model


def diarize_audio_batch(model: Any, audios: list[Audio], settings: SortformerSettings) -> list[list[DiarizationSegment]]:
    if not audios:
        return []
    wavs = [sortformer_audio_array(audio, settings.sample_rate) for audio in audios]
    batch_size = min(settings.batch_size, len(wavs))
    outputs = model.diarize(audio=wavs, batch_size=batch_size, sample_rate=settings.sample_rate)
    if len(outputs) != len(audios):
        raise ValueError("sortformer_output_count_mismatch")
    return [parse_sortformer_segments(output) for output in outputs]


def sortformer_audio_array(audio: Audio, sample_rate: int):
    librosa, np = _audio_processing_dependencies()
    y, _sr = librosa.load(BytesIO(audio.data), sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def parse_sortformer_segments(output: Any) -> list[DiarizationSegment]:
    if isinstance(output, str):
        return [_parse_sortformer_line(line) for line in output.splitlines() if line.strip()]
    if isinstance(output, list):
        return [_parse_sortformer_item(item) for item in output]
    raise ValueError(f"unsupported Sortformer diarization output: {type(output).__name__}")


def speaker_audio_segments(audio: Audio, segments: list[DiarizationSegment], settings: SortformerSettings) -> list[Audio]:
    info = wav_info(audio.data)
    outputs = []
    for index, segment in enumerate(segments):
        start = max(0.0, float(segment.start))
        end = min(audio.duration, float(segment.end))
        if end - start < settings.min_segment_sec:
            continue
        outputs.append(_speaker_audio(audio, segment.speaker, start, end, info, index))
    return outputs


def cut_audio_by_segments(audio: Audio, settings: CutAudioSettings) -> list[Audio]:
    assert audio.data is not None, f"audio bytes are required: {audio.id}"
    assert audio.segments, f"audio segments are required: {audio.id}"
    return [cut_audio_by_segment(audio, segment, settings, index) for index, segment in enumerate(audio.segments)]


def cut_audio_by_segment(audio: Audio, segment: AudioSegment, settings: CutAudioSettings, index: int = 0) -> Audio:
    del settings
    local_start = max(0.0, float(segment.start) - audio.start)
    local_end = max(local_start, segment.end - audio.start)
    assert local_end <= audio.duration + 1e-6, f"segment outside audio bounds: {segment.id}"
    info = wav_info(audio.data)
    data = extract_wav_range(audio.data, local_start, local_end, info)
    duration = max(0.0, local_end - local_start)
    cut_id = stable_id("audio", audio.audio_file_id, audio.id, segment.id, index, local_start, local_end)
    relative_segment = replace(
        segment,
        start=0.0,
        end=duration,
        metadata={
            **segment.metadata,
            "source_start": segment.start,
            "source_end": segment.end,
        },
    )
    return Audio(
        audio_file_id=audio.audio_file_id,
        name=segment.name or audio.name,
        data=data,
        sample_rate=int(info["sample_rate"]),
        channels=int(info["channels"]),
        start=0.0,
        end=duration,
        confidence=segment.confidence if segment.confidence is not None else audio.confidence,
        id=cut_id,
        lineage_id=stable_id("lineage", audio.lineage_id, segment.lineage_id),
        metadata={
            **audio.metadata,
            **segment.metadata,
            "source_audio_id": str(audio.audio_file_id),
            "source_audio_node_id": audio.id,
            "source_segment_id": segment.id,
            "source_segment_entry_id": segment.segment_id,
            "source_start": segment.start,
            "source_end": segment.end,
            "text": segment.text,
            "phon": segment.phon,
            "speaker": segment.speaker,
            "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        },
        byte_length=len(data),
        virtual=audio.virtual,
        segments=[relative_segment],
    )


def _speaker_audio(audio: Audio, speaker: str, start: float, end: float, info: dict[str, int], index: int) -> Audio:
    absolute_start = audio.start + start
    absolute_end = audio.start + end
    speaker_id = stable_id("speaker", audio.audio_file_id, audio.id, speaker)
    audio_id = stable_id("audio", audio.audio_file_id, audio.id, speaker, index, absolute_start, absolute_end)
    data = extract_wav_range(audio.data, start, end, info)
    return Audio(
        audio.audio_file_id,
        audio.name,
        data,
        int(info["sample_rate"]),
        int(info["channels"]),
        absolute_start,
        absolute_end,
        audio.confidence,
        audio_id,
        stable_id("lineage", audio.lineage_id, audio_id),
        {
            **audio.metadata,
            "diarization": {
                "model": "sortformer",
                "speaker": speaker,
                "speaker_id": speaker_id,
                "segment_index": index,
                "start": absolute_start,
                "end": absolute_end,
            },
            "speaker": speaker,
            "speaker_id": speaker_id,
        },
    )


def _segment_audio(audio: Audio, start: float, end: float, info: dict[str, int], index: int) -> Audio:
    absolute_start = audio.start + start
    absolute_end = audio.start + end
    segment_id = stable_id("audio", audio.audio_file_id, audio.id, index, absolute_start, absolute_end)
    data = extract_wav_range(audio.data, start, end, info)
    return Audio(
        audio.audio_file_id,
        audio.name,
        data,
        int(info["sample_rate"]),
        int(info["channels"]),
        absolute_start,
        absolute_end,
        audio.confidence,
        segment_id,
        stable_id("lineage", audio.lineage_id, segment_id),
        {
            **audio.metadata,
            "vad": {
                "source_audio_id": audio.id,
                "segment_index": index,
                "start": absolute_start,
                "end": absolute_end,
            },
        },
    )


def _merge_spans(spans: list[tuple[float, float]], max_gap_seconds: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= max_gap_seconds:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_long_spans(spans: list[tuple[float, float]], max_segment_sec: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    for start, end in spans:
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_segment_sec)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
    return chunks


def _sortformer_device(torch: Any, setting: str) -> str:
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("sortformer_cuda_not_available")
    return setting


def _configure_sortformer_modules(model: Any, settings: SortformerSettings) -> None:
    modules = model.sortformer_modules
    if settings.chunk_len is not None:
        modules.chunk_len = settings.chunk_len
    if settings.chunk_right_context is not None:
        modules.chunk_right_context = settings.chunk_right_context
    if settings.fifo_len is not None:
        modules.fifo_len = settings.fifo_len
    if settings.spkcache_update_period is not None:
        modules.spkcache_update_period = settings.spkcache_update_period


def _parse_sortformer_line(line: str) -> DiarizationSegment:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"invalid Sortformer diarization line: {line}")
    return DiarizationSegment(float(parts[0]), float(parts[1]), str(parts[2]))


def _parse_sortformer_item(item: Any) -> DiarizationSegment:
    if isinstance(item, str):
        return _parse_sortformer_line(item)
    if isinstance(item, dict):
        speaker = item.get("speaker", item.get("label"))
        if speaker is None:
            raise ValueError("Sortformer diarization item missing speaker")
        return DiarizationSegment(float(item["start"]), float(item["end"]), str(speaker))
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        return DiarizationSegment(float(item[0]), float(item[1]), str(item[2]))
    raise ValueError(f"invalid Sortformer diarization item: {item!r}")


def _audio_processing_dependencies():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError("VadDetect requires optional audio dependencies 'librosa' and 'numpy'") from exc
    return librosa, np
