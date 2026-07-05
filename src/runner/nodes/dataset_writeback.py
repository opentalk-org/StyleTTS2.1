from __future__ import annotations

from uuid import UUID

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import AUDIO_REF, JSON
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.datasets import crud as dataset_crud


class DatasetWritebackSettings(StrictSettings):
    dataset_id: UUID


class AssignVoiceSettings(StrictSettings):
    voice: str


class AddAudioToDatasetNode(Node):
    NODE_TYPE = "AddAudioToDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.add_audio_file_to_dataset(session, self.settings.dataset_id, inputs["audio_ref"].audio_file_id)
        return [{"writeback_result": {"updated": inputs["audio_ref"].name}} for inputs in batch]


class RemoveAudioFromDatasetNode(Node):
    NODE_TYPE = "RemoveAudioFromDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.remove_audio_file_from_dataset(session, self.settings.dataset_id, inputs["audio_ref"].audio_file_id)
        return [{"writeback_result": {"updated": inputs["audio_ref"].name}} for inputs in batch]


class AssignVoiceNode(Node):
    NODE_TYPE = "AssignVoice"
    CATEGORY = "Dataset"
    SETTINGS = AssignVoiceSettings
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        return [{"writeback_result": {"audio_file_id": str(inputs["audio_ref"].audio_file_id), "voice": self.settings.voice}} for inputs in batch]


class DeleteAudioRecordsNode(Node):
    NODE_TYPE = "DeleteAudioRecords"
    CATEGORY = "Dataset"
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF)}
    OUTPUTS = {"writeback_result": Port("writeback_result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                audio_crud.delete_audio_file(session, inputs["audio_ref"].audio_file_id)
        return [{"writeback_result": {"deleted": str(inputs["audio_ref"].audio_file_id)}} for inputs in batch]
