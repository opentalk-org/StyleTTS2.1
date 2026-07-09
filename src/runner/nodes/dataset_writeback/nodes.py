from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, JsonPort
from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioUpdate
from shared.db.datasets import crud as dataset_crud
from shared.db.voices import crud as voice_crud


class DatasetWritebackSettings(StrictSettings):
    dataset_id: UUID


class AssignVoiceSettings(StrictSettings):
    voice: str = Field(title="Voice")


class AddAudioToDatasetNode(Node):
    NODE_TYPE = "AddAudioToDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=128, max_size=512)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios = [inputs["audio"] for inputs in batch]
        for _ in context.cancellable(audios):
            pass
        with database_session() as session:
            dataset_crud.bulk_add_audio_files_to_dataset(session, self.settings.dataset_id, [audio.audio_file_id for audio in audios])
        return [{"writeback_result": {"updated": audio.name}} for audio in audios]


class RemoveAudioFromDatasetNode(Node):
    NODE_TYPE = "RemoveAudioFromDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=128, max_size=512)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                context.check_cancel()
                dataset_crud.remove_audio_file_from_dataset(session, self.settings.dataset_id, inputs["audio"].audio_file_id)
        return [{"writeback_result": {"updated": inputs["audio"].name}} for inputs in batch]


class AssignVoiceNode(Node):
    NODE_TYPE = "AssignVoice"
    CATEGORY = "Dataset"
    SETTINGS = AssignVoiceSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            assignment = _voice_assignment(session, self.settings.voice)
            for inputs in batch:
                context.check_cancel()
                audio: Audio = inputs["audio"]
                item = audio_crud.get_audio_file(session, audio.audio_file_id)
                metadata = _assigned_metadata(item.metadata_, assignment)
                segments = [_assigned_segment(segment, assignment) for segment in item.segments]
                updated = audio_crud.update_audio_file(
                    session,
                    audio.audio_file_id,
                    AudioUpdate(
                        name=item.name,
                        wav_bytes=None,
                        duration=item.duration,
                        segments=segments,
                        metadata=metadata,
                        virtual=item.virtual,
                    ),
                )
                context.check_cancel()
                outputs.append({
                    "audio": replace(
                        audio,
                        name=updated.name,
                        metadata=updated.metadata_,
                        segments=[replace(segment, speaker=assignment.speaker, voice_id=assignment.voice_id) for segment in audio.segments],
                        virtual=updated.virtual,
                    ),
                    "writeback_result": {
                        "audio_file_id": str(updated.id),
                        "speaker": assignment.speaker,
                        "voice_id": str(assignment.voice_id) if assignment.voice_id is not None else None,
                    },
                })
        return outputs


class DeleteAudioRecordsNode(Node):
    NODE_TYPE = "DeleteAudioRecords"
    CATEGORY = "Dataset"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                context.check_cancel()
                audio_crud.delete_audio_file(session, inputs["audio"].audio_file_id)
        return [{"writeback_result": {"deleted": str(inputs["audio"].audio_file_id)}} for inputs in batch]


@dataclass(frozen=True)
class VoiceAssignment:
    speaker: str
    voice_id: UUID | None


def _voice_assignment(session, value: str) -> VoiceAssignment:
    voice = value.strip()
    if not voice:
        raise ValueError("AssignVoice requires a voice")
    voice_id = _parse_uuid(voice)
    if voice_id is None:
        return VoiceAssignment(speaker=voice, voice_id=None)
    for item in voice_crud.list_voices(session):
        if item.id == voice_id:
            return VoiceAssignment(speaker=item.name, voice_id=item.id)
    raise KeyError(f"Voice not found: {voice_id}")


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


def _assigned_metadata(metadata: dict, assignment: VoiceAssignment) -> dict:
    out = {**metadata, "speaker": assignment.speaker}
    if assignment.voice_id is None:
        out.pop("voice_id", None)
    else:
        out["voice_id"] = str(assignment.voice_id)
    return out


def _assigned_segment(segment: dict, assignment: VoiceAssignment) -> dict:
    metadata = dict(segment["metadata"]) if isinstance(segment.get("metadata"), dict) else {}
    metadata = _assigned_metadata(metadata, assignment)
    return {
        **segment,
        "speaker": assignment.speaker,
        "voice_id": str(assignment.voice_id) if assignment.voice_id is not None else None,
        "metadata": metadata,
    }
