from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import AUDIO, JSON
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
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.add_audio_file_to_dataset(session, self.settings.dataset_id, inputs["audio"].audio_file_id)
        return [{"writeback_result": {"updated": inputs["audio"].name}} for inputs in batch]


class RemoveAudioFromDatasetNode(Node):
    NODE_TYPE = "RemoveAudioFromDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.remove_audio_file_from_dataset(session, self.settings.dataset_id, inputs["audio"].audio_file_id)
        return [{"writeback_result": {"updated": inputs["audio"].name}} for inputs in batch]


class AssignVoiceNode(Node):
    NODE_TYPE = "AssignVoice"
    CATEGORY = "Dataset"
    SETTINGS = AssignVoiceSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO), "writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            assignment = _voice_assignment(session, self.settings.voice)
            for inputs in batch:
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
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
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
