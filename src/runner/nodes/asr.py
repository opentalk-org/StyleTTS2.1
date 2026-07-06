from __future__ import annotations

from typing import Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO, TRANSCRIPT
from runner.nodes.models import Audio, Transcript, stable_id


class TranscribeSettings(StrictSettings):
    language: str = "auto"
    batch_size: int = Field(default=16, ge=1, le=128)
    scope: Literal["full", "segments"] = "full"
    segment_mode: Literal["replace", "add"] = "replace"
    segment_batch_size: int = Field(default=16, ge=1, le=128)


class TranscribeNode(Node):
    CATEGORY = "Audio / ASR"
    MODEL_NAME = "asr"
    SETTINGS = TranscribeSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=False, exclusive_group="accelerator")

    async def execute(self, batch, context):
        outputs = []
        for index, inputs in enumerate(batch, start=1):
            audio = inputs["audio"]
            await context.report_progress(self.id, index, len(batch), f"{self.MODEL_NAME} transcribed {index}/{len(batch)}")
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            transcript_id = stable_id("transcript", self.MODEL_NAME, audio.audio_file_id, audio.id)
            segment_id = stable_id("transcript_segment", transcript_id, audio.start, audio.end, 0)
            text = f"[{self.MODEL_NAME}] transcript for {audio.audio_file_id}"
            transcript = Transcript(
                text=text,
                model=self.MODEL_NAME,
                source_audio_id=audio.audio_file_id,
                start=audio.start,
                end=audio.end,
                speaker=getattr(audio, "speaker", None),
                id=transcript_id,
                lineage_id=stable_id("lineage", self.MODEL_NAME, audio.audio_file_id),
                segments=[{
                    "id": segment_id,
                    "start": audio.start,
                    "end": audio.end,
                    "text": text,
                    "phon": "",
                    "speaker": getattr(audio, "speaker", None) or "",
                }],
                metadata={
                    **audio.metadata,
                    "language": self.settings.language,
                    "sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                    "scope": self.settings.scope,
                    "segment_mode": self.settings.segment_mode,
                    "segment_batch_size": self.settings.segment_batch_size,
                },
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
