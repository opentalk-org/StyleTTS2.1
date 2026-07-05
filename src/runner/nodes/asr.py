from __future__ import annotations

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO_LIKE, TRANSCRIPT
from runner.nodes.models import Transcript, stable_id


class TranscribeSettings(StrictSettings):
    language: str = "auto"
    beam_size: int = Field(default=5, ge=1, le=16)
    batch_size: int = Field(default=16, ge=1, le=128)


class TranscribeNode(Node):
    CATEGORY = "Audio / ASR"
    MODEL_NAME = "asr"
    SETTINGS = TranscribeSettings
    INPUTS = {"audio": Port("audio", AUDIO_LIKE)}
    OUTPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=False, exclusive_group="accelerator")

    async def execute(self, batch, context):
        outputs = []
        for index, inputs in enumerate(batch, start=1):
            audio = inputs["audio"]
            await context.report_progress(self.id, index, len(batch), f"{self.MODEL_NAME} transcribed {index}/{len(batch)}")
            source_audio_id = getattr(audio, "source_audio_id", audio.audio_file_id)
            transcript_id = stable_id("transcript", self.MODEL_NAME, source_audio_id, getattr(audio, "id", "audio"))
            start = getattr(audio, "start", None)
            end = getattr(audio, "end", None)
            text = f"[{self.MODEL_NAME}] transcript for {source_audio_id}"
            transcript = Transcript(
                text=text,
                model=self.MODEL_NAME,
                source_audio_id=source_audio_id,
                start=start,
                end=end,
                speaker=getattr(audio, "speaker", None),
                id=transcript_id,
                lineage_id=stable_id("lineage", self.MODEL_NAME, source_audio_id),
                segments=[{"start": start, "end": end, "text": text}],
                metadata={**audio.metadata, "language": self.settings.language, "beam_size": self.settings.beam_size},
            )
            outputs.append({"transcript": transcript})
        return outputs


class WhisperTranscribeNode(TranscribeNode):
    NODE_TYPE = "WhisperTranscribe"
    MODEL_NAME = "whisper"


class CanaryTranscribeNode(TranscribeNode):
    NODE_TYPE = "CanaryTranscribe"
    MODEL_NAME = "canary"


class ParakeetTranscribeNode(TranscribeNode):
    NODE_TYPE = "ParakeetTranscribe"
    MODEL_NAME = "parakeet"
