from __future__ import annotations

import json
from dataclasses import replace

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, SaveResult, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class LoadAudioSettings(StrictSettings):
    sample_rate: int = Field(default=24000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)


class SaveAudioArtifactSettings(StrictSettings):
    output_subdir: str = "audio"
    extension: str = "wav"


class LoadAudioNode(Node):
    NODE_TYPE = "LoadAudio"
    CATEGORY = "Audio"
    SETTINGS = LoadAudioSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                data = audio.data if audio.data is not None else audio_crud.read_audio_file(session, audio.audio_file_id)
                loaded = replace(
                    audio,
                    data=data,
                    sample_rate=self.settings.sample_rate,
                    channels=self.settings.channels,
                    metadata={**audio.metadata, "byte_length": len(data), "source_duration": audio.duration},
                    byte_length=len(data),
                )
                outputs.append({"audio": loaded})
        return outputs


class SaveAudioArtifactNode(Node):
    NODE_TYPE = "SaveAudioArtifact"
    CATEGORY = "Audio"
    SETTINGS = SaveAudioArtifactSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"save_result": SaveResultPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        out_dir = context.output_dir / self.settings.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert audio.data is not None, f"audio bytes are required: {audio.id}"
            out_path = out_dir / f"{audio.id}.{self.settings.extension}"
            kind = "audio"
            if self.settings.extension == "json":
                kind = "audio_segment" if audio.start > 0.0 or audio.end < audio.metadata["source_duration"] else "audio"
                out_path.write_text(json.dumps({
                    "audio_file_id": str(audio.audio_file_id),
                    "name": audio.name,
                    "start": audio.start,
                    "end": audio.end,
                    "confidence": audio.confidence,
                    "sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                    "metadata": audio.metadata,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                out_path.write_bytes(audio.data)
            result_id = stable_id("save", out_path)
            outputs.append({"save_result": SaveResult(out_path, kind, result_id, audio.lineage_id, audio.metadata)})
        return outputs
