from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.asr.audio import extract_wav_range, wav_info
from runner.nodes.datatypes import AUDIO, JSON
from runner.nodes.models import Audio, stable_id
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode, AudioFeatureSettings, analyze_audio_features


class VadSettings(StrictSettings):
    min_segment_sec: float = Field(default=1.0, ge=0.1, le=30.0)
    max_segment_sec: float = Field(default=12.0, ge=1.0, le=60.0)
    padding_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    max_silence_gap_ms: int = Field(default=400, ge=50, le=3000)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    hop_length: int = Field(default=512, ge=64, le=4096)


class CutAudioSettings(StrictSettings):
    fade_ms: int = Field(default=0, ge=0, le=100)


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


class CutAudioBySegmentsNode(Node):
    NODE_TYPE = "CutAudioBySegments"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = CutAudioSettings
    INPUTS = {"audio": Port("audio", AUDIO), "segment": Port("segment", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            segment = inputs["segment"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            assert isinstance(segment, Audio), f"unsupported segment input: {type(segment).__name__}"
            outputs.append({"audio": cut_audio_by_segment(audio, segment, self.settings)})
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


def cut_audio_by_segment(audio: Audio, segment: Audio, settings: CutAudioSettings) -> Audio:
    del settings
    local_start = max(0.0, segment.start - audio.start)
    local_end = max(local_start, segment.end - audio.start)
    assert local_end <= audio.duration + 1e-6, f"segment outside audio bounds: {segment.id}"
    info = wav_info(audio.data)
    data = extract_wav_range(audio.data, local_start, local_end, info)
    segment_id = stable_id("audio", audio.audio_file_id, audio.id, segment.id, local_start, local_end)
    return replace(
        segment,
        audio_file_id=audio.audio_file_id,
        data=data,
        sample_rate=int(info["sample_rate"]),
        channels=int(info["channels"]),
        id=segment_id,
        lineage_id=stable_id("lineage", audio.lineage_id, segment.lineage_id),
        metadata={
            **audio.metadata,
            **segment.metadata,
            "source_audio_id": str(audio.audio_file_id),
            "source_audio_node_id": audio.id,
            "source_segment_id": segment.id,
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


def _audio_processing_dependencies():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError("VadDetect requires optional audio dependencies 'librosa' and 'numpy'") from exc
    return librosa, np
