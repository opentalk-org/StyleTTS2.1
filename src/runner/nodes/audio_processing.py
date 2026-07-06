from __future__ import annotations

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.audio_enhancement.denoise import DeepFilterNetDenoiseNode, DeepFilterNetSettings
from runner.nodes.audio_enhancement.normalize import NormalizeLoudnessNode, NormalizeSettings
from runner.nodes.datatypes import AUDIO, JSON
from runner.nodes.models import Audio, stable_id
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode, AudioFeatureSettings, analyze_audio_features


class VadSettings(StrictSettings):
    min_segment_sec: float = Field(default=1.0, ge=0.1, le=30.0)
    max_segment_sec: float = Field(default=12.0, ge=1.0, le=60.0)
    padding_sec: float = Field(default=0.12, ge=0.0, le=1.0)
    max_silence_gap_ms: int = Field(default=400, ge=50, le=3000)


class CutAudioSettings(StrictSettings):
    fade_ms: int = Field(default=0, ge=0, le=100)


class VadDetectNode(Node):
    NODE_TYPE = "VadDetect"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = VadSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            start = audio.start
            end = start + min(audio.duration, self.settings.max_segment_sec)
            segment_id = stable_id("audio", audio.audio_file_id, start, end)
            outputs.append({"audio": Audio(audio.audio_file_id, audio.name, audio.data, audio.sample_rate, audio.channels, start, end, 0.9, segment_id, audio.lineage_id, audio.metadata)})
        return outputs


class CutAudioBySegmentsNode(Node):
    NODE_TYPE = "CutAudioBySegments"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = CutAudioSettings
    INPUTS = {"audio": Port("audio", AUDIO), "segment": Port("segment", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}

    async def execute(self, batch, context):
        return [{"audio": inputs["segment"]} for inputs in batch]


class SortformerDiarizationNode(Node):
    NODE_TYPE = "SortformerDiarization"
    CATEGORY = "Audio / Segmentation"
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"turns": Port("turns", JSON)}

    async def execute(self, batch, context):
        return [{"turns": {"speaker": "speaker_0", "confidence": 0.8, "source": getattr(inputs["audio"], "id", "")}} for inputs in batch]


class CutAudioBySpeakersNode(Node):
    NODE_TYPE = "CutAudioBySpeakers"
    CATEGORY = "Audio / Segmentation"
    INPUTS = {"audio": Port("audio", AUDIO), "turns": Port("turns", JSON)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            outputs.append({"audio": audio})
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
