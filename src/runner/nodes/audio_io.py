from __future__ import annotations

import json

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO_REF, BUCKET_AUDIO, SAVE_RESULT, TRANSCRIPT
from runner.nodes.models import BucketAudio, SaveResult, Transcript, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class LoadBucketAudioSettings(StrictSettings):
    sample_rate: int = Field(default=24000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)


class SaveTranscriptSettings(StrictSettings):
    format: str = "json"
    overwrite: bool = True
    output_subdir: str = "transcripts"


class SaveAudioArtifactSettings(StrictSettings):
    output_subdir: str = "audio"
    extension: str = "wav"


class LoadBucketAudioNode(Node):
    NODE_TYPE = "LoadBucketAudio"
    CATEGORY = "Audio / IO"
    SETTINGS = LoadBucketAudioSettings
    INPUTS = {"audio": Port("audio", AUDIO_REF)}
    OUTPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                ref = inputs["audio"]
                data = audio_crud.read_audio_file(session, ref.audio_file_id)
                audio_id = stable_id("audio", ref.audio_file_id, ref.name)
                audio = BucketAudio(
                    audio_file_id=ref.audio_file_id,
                    name=ref.name,
                    data=data,
                    sample_rate=self.settings.sample_rate,
                    channels=self.settings.channels,
                    duration=ref.duration,
                    id=audio_id,
                    lineage_id=audio_id,
                    metadata={**ref.metadata, "byte_length": ref.byte_length},
                )
                outputs.append({"audio": audio})
        return outputs


class SaveTranscriptNode(Node):
    NODE_TYPE = "SaveTranscript"
    CATEGORY = "Audio / IO"
    SETTINGS = SaveTranscriptSettings
    INPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    OUTPUTS = {"result": Port("result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        out_dir = context.output_dir / self.settings.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for inputs in batch:
            transcript: Transcript = inputs["transcript"]
            out_path = out_dir / f"{transcript.id}.{self.settings.format}"
            assert self.settings.overwrite or not out_path.exists(), f"transcript exists: {out_path}"
            payload = {
                "id": transcript.id,
                "model": transcript.model,
                "text": transcript.text,
                "source_audio_id": str(transcript.source_audio_id),
                "start": transcript.start,
                "end": transcript.end,
                "speaker": transcript.speaker,
                "segments": transcript.segments,
                "metadata": transcript.metadata,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result_id = stable_id("save", out_path)
            outputs.append({"result": SaveResult(out_path, "transcript", result_id, transcript.lineage_id, payload["metadata"])})
        return outputs


class SaveAudioArtifactNode(Node):
    NODE_TYPE = "SaveAudioArtifact"
    CATEGORY = "Audio / IO"
    SETTINGS = SaveAudioArtifactSettings
    INPUTS = {"audio": Port("audio", BUCKET_AUDIO)}
    OUTPUTS = {"result": Port("result", SAVE_RESULT)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        out_dir = context.output_dir / self.settings.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            out_path = out_dir / f"{audio.id}.{self.settings.extension}"
            out_path.write_bytes(audio.data)
            result_id = stable_id("save", out_path)
            outputs.append({"result": SaveResult(out_path, "audio", result_id, audio.lineage_id, audio.metadata)})
        return outputs
