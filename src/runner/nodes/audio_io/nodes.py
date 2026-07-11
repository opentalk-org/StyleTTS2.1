from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, SaveResult, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFileCreate
from shared.db.audio import crud as audio_crud


class LoadAudioSettings(StrictSettings):
    sample_rate: int = Field(default=24000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)


class SaveAudioArtifactSettings(StrictSettings):
    output_subdir: str = "audio"
    extension: str = "wav"


def _artifact_bytes(audio: Audio, extension: str) -> tuple[bytes, str, str]:
    if extension == "json":
        kind = "audio_segment" if audio.start > 0.0 or audio.end < audio.metadata["source_duration"] else "audio"
        payload = json.dumps({
            "audio_file_id": str(audio.audio_file_id),
            "name": audio.name,
            "start": audio.start,
            "end": audio.end,
            "confidence": audio.confidence,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "metadata": audio.metadata,
        }, ensure_ascii=False, indent=2).encode("utf-8")
        return payload, kind, "application/json"
    assert audio.data is not None, f"audio bytes are required: {audio.id}"
    return audio.data, "audio", f"audio/{extension}"


class LoadAudioNode(Node):
    NODE_TYPE = "LoadAudio"
    DESCRIPTION = "Load the raw audio bytes for each incoming audio item, reading from the database when they are not already present, and normalize the sample rate and channel count. Takes audio references and outputs audio with decoded data attached, ready for downstream processing. Use it at the start of a pipeline to bring audio into memory. Set the target sample rate and channels."
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
    DESCRIPTION = "Store each audio item as a durable artifact in the object bucket and emit a save result referencing it. Takes audio and outputs a save result with the bucket object key and metadata. Choose the output subfolder (used to name and group the artifact) and file extension: use an audio extension like wav to save the sound, or json to save just the audio's metadata and timing instead. Use it at the end of a pipeline to persist results."
    CATEGORY = "Audio"
    SETTINGS = SaveAudioArtifactSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"save_result": SaveResultPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio = inputs["audio"]
                data, kind, content_type = _artifact_bytes(audio, self.settings.extension)
                name = f"{self.settings.output_subdir}/{audio.id}.{self.settings.extension}"
                metadata = {**audio.metadata, "content_type": content_type, "subdir": self.settings.output_subdir}
                artifact = asset_crud.create_extra_file(
                    session,
                    ExtraFileCreate(name=name, data=data, type_="artifact", metadata=metadata),
                )
                result_id = stable_id("save", artifact.path)
                result_metadata = {**metadata, "artifact_id": str(artifact.id), "bucket_key": artifact.path}
                outputs.append({"save_result": SaveResult(Path(artifact.path), kind, result_id, audio.lineage_id, result_metadata)})
        return outputs
