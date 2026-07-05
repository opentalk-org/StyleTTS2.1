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
    voice_id: UUID


class AddAudioToDatasetNode(Node):
    NODE_TYPE = "AddAudioToDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio": Port("audio", AUDIO_REF)}
    OUTPUTS = {"result": Port("result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.add_audio_file_to_dataset(session, self.settings.dataset_id, inputs["audio"].audio_file_id)
        return [{"result": {"updated": inputs["audio"].name}} for inputs in batch]


class RemoveAudioFromDatasetNode(Node):
    NODE_TYPE = "RemoveAudioFromDataset"
    CATEGORY = "Dataset"
    SETTINGS = DatasetWritebackSettings
    INPUTS = {"audio": Port("audio", AUDIO_REF)}
    OUTPUTS = {"result": Port("result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                dataset_crud.remove_audio_file_from_dataset(session, self.settings.dataset_id, inputs["audio"].audio_file_id)
        return [{"result": {"updated": inputs["audio"].name}} for inputs in batch]


class AssignVoiceNode(Node):
    NODE_TYPE = "AssignVoice"
    CATEGORY = "Dataset"
    SETTINGS = AssignVoiceSettings
    INPUTS = {"audio": Port("audio", AUDIO_REF)}
    OUTPUTS = {"result": Port("result", JSON)}

    async def execute(self, batch, context):
        return [{"result": {"audio_file_id": str(inputs["audio"].audio_file_id), "voice_id": str(self.settings.voice_id)}} for inputs in batch]


class DeleteAudioRecordsNode(Node):
    NODE_TYPE = "DeleteAudioRecords"
    CATEGORY = "Dataset"
    INPUTS = {"audio": Port("audio", AUDIO_REF)}
    OUTPUTS = {"result": Port("result", JSON)}

    async def execute(self, batch, context):
        with database_session() as session:
            for inputs in batch:
                audio_crud.delete_audio_file(session, inputs["audio"].audio_file_id)
        return [{"result": {"deleted": str(inputs["audio"].audio_file_id)}} for inputs in batch]
