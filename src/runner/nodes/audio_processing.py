from __future__ import annotations

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO_LIKE, AUDIO_SEGMENT, BUCKET_AUDIO, JSON
from runner.nodes.models import AudioSegment, BucketAudio, stable_id


class VadSettings(StrictSettings):
    min_speech_sec: float = Field(default=0.25, ge=0.05)
    max_segment_sec: float = Field(default=30.0, ge=1.0)
    padding_sec: float = Field(default=0.1, ge=0.0)


class NormalizeSettings(StrictSettings):
    target_lufs: float = -23.0
    peak_limit: float = Field(default=0.95, gt=0.0, le=1.0)


class DeepFilterNetSettings(StrictSettings):
    attenuation_db: float = Field(default=12.0, ge=0.0, le=60.0)


class VadDetectNode(Node):
    NODE_TYPE = "VadDetect"
    CATEGORY = "Audio / Segmentation"
    SETTINGS = VadSettings
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    OUTPUTS = {"segment": Port("segment", AUDIO_SEGMENT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            end = min(audio.duration, self.settings.max_segment_sec)
            segment_id = stable_id("segment", audio.audio_file_id, 0, end)
            outputs.append({"segment": AudioSegment(audio.audio_file_id, 0.0, end, 0.9, segment_id, audio.lineage_id, audio.metadata)})
        return outputs


class CutAudioBySegmentsNode(Node):
    NODE_TYPE = "CutAudioBySegments"
    CATEGORY = "Audio / Segmentation"
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO), "segment": Port("segment", AUDIO_SEGMENT)}
    OUTPUTS = {"audio": Port("audio", AUDIO_SEGMENT)}

    async def execute(self, batch, context):
        return [{"audio": inputs["segment"]} for inputs in batch]


class SortformerDiarizationNode(Node):
    NODE_TYPE = "SortformerDiarization"
    CATEGORY = "Audio / Segmentation"
    INPUTS = {"audio": Port("audio", AUDIO_LIKE)}
    OUTPUTS = {"turns": Port("turns", JSON)}

    async def execute(self, batch, context):
        return [{"turns": {"speaker": "speaker_0", "confidence": 0.8, "source": getattr(inputs["audio"], "id", "")}} for inputs in batch]


class CutAudioBySpeakersNode(Node):
    NODE_TYPE = "CutAudioBySpeakers"
    CATEGORY = "Audio / Segmentation"
    INPUTS = {"audio": Port("audio", AUDIO_SEGMENT), "turns": Port("turns", JSON)}
    OUTPUTS = {"audio": Port("audio", AUDIO_SEGMENT)}

    async def execute(self, batch, context):
        return [{"audio": inputs["audio"]} for inputs in batch]


class DeepFilterNetDenoiseNode(Node):
    NODE_TYPE = "DeepFilterNetDenoise"
    CATEGORY = "Audio / Enhancement"
    SETTINGS = DeepFilterNetSettings
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    OUTPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 4})

    async def execute(self, batch, context):
        return [{"audio": inputs["audio"]} for inputs in batch]


class NormalizeLoudnessNode(Node):
    NODE_TYPE = "NormalizeLoudness"
    CATEGORY = "Audio / Enhancement"
    SETTINGS = NormalizeSettings
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    OUTPUTS = {"audio": Port("audio", BUCKET_AUDIO)}

    async def execute(self, batch, context):
        return [{"audio": inputs["audio"]} for inputs in batch]


class CalculateAudioStatsNode(Node):
    NODE_TYPE = "CalculateAudioStats"
    CATEGORY = "Audio / Statistics"
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    OUTPUTS = {"stats": Port("stats", JSON)}

    async def execute(self, batch, context):
        return [{"stats": {"duration": inputs["audio"].duration, "bytes": len(inputs["audio"].data)}} for inputs in batch]
