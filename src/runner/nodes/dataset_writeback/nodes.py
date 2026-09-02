from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, JsonPort
from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioUpdate
from shared.db.datasets import crud as dataset_crud
from shared.audio_annotations import AudioAnnotations


class DatasetWritebackSettings(StrictSettings):
    dataset_id: UUID


class AssignSpeakerSettings(StrictSettings):
    speaker_id: str


class AddAudioToDatasetNode(Node):
    NODE_TYPE = "AddAudioToDataset"
    DESCRIPTION = "Add each incoming audio item to a chosen dataset, persisting the membership in the database. Takes audio in and passes through a small JSON result recording which items were updated. Use it near the end of a workflow to collect processed audio into a dataset."
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
        dataset_crud.bulk_add_audio_files_to_dataset(
            self.settings.dataset_id,
            [audio.audio_file_id for audio in audios],
        )
        return [{"writeback_result": {"updated": audio.name}} for audio in audios]


class RemoveAudioFromDatasetNode(Node):
    NODE_TYPE = "RemoveAudioFromDataset"
    DESCRIPTION = "Remove each incoming audio item from a chosen dataset, updating the membership in the database. Takes audio in and emits a small JSON result noting which items were updated. Use it to prune audio out of a dataset without deleting the underlying files."
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
        dataset_crud.bulk_remove_audio_files_from_dataset(
            self.settings.dataset_id,
            [audio.audio_file_id for audio in audios],
        )
        return [{"writeback_result": {"updated": audio.name}} for audio in audios]


class AssignSpeakerNode(Node):
    NODE_TYPE = "AssignSpeaker"
    DESCRIPTION = "Assign a speaker ID to each incoming audio item and its segments, saving the change to the database."
    CATEGORY = "Dataset"
    SETTINGS = AssignSpeakerSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        with database_session() as session:
            speaker_id = self.settings.speaker_id.strip()
            if not speaker_id:
                raise ValueError("AssignSpeaker requires speaker_id")
            items = audio_crud.get_audio_files_bulk(
                session,
                [audio.audio_file_id for audio in audios],
            )
            payloads = {}
            for audio in audios:
                context.check_cancel()
                item = items[audio.audio_file_id]
                stored_segments = audio_crud.list_audio_segments_bulk(
                    session, [item.id]
                )[item.id]
                segments = [
                    _assigned_segment(segment, speaker_id)
                    for segment in stored_segments
                ]
                payloads[audio.audio_file_id] = AudioUpdate(
                    name=item.name,
                    wav_bytes=None,
                    duration=item.duration,
                    segments=segments,
                    annotations=audio_crud.audio_file_annotations(item).model_copy(
                        update={
                            "speaker_id": speaker_id,
                        }
                    ),
                    language=item.language,
                    style_prompt=item.style_prompt,
                    voice_prompt=item.voice_prompt,
                    virtual=item.virtual,
                )
            updated_by_id = audio_crud.bulk_update_audio_files(session, payloads)
        outputs = []
        for audio in audios:
            context.check_cancel()
            updated = updated_by_id[audio.audio_file_id]
            outputs.append(
                {
                    "audio": replace(
                        audio,
                        name=updated.name,
                        annotations=audio_crud.audio_file_annotations(updated),
                        segments=[
                            replace(
                                segment,
                                annotations=segment.annotations.model_copy(
                                    update={
                                        "speaker_id": speaker_id,
                                    }
                                ),
                            )
                            for segment in audio.segments
                        ],
                        virtual=updated.virtual,
                    ),
                    "writeback_result": {
                        "audio_file_id": str(updated.id),
                        "speaker_id": speaker_id,
                    },
                }
            )
        return outputs


class DeleteAudioRecordsNode(Node):
    NODE_TYPE = "DeleteAudioRecords"
    DESCRIPTION = "Permanently delete each incoming audio item and its record from the database. Takes audio in and emits a JSON result listing the deleted IDs. Use it to clean up unwanted audio; this is destructive and cannot be undone."
    CATEGORY = "Dataset"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH, preferred_size=256, max_size=256, timeout_ms=20
    )
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 512

    async def execute(self, batch, context):
        audios = [inputs["audio"] for inputs in batch]
        for _ in context.cancellable(audios):
            pass
        with database_session() as session:
            audio_crud.bulk_delete_audio_files(
                session,
                [audio.audio_file_id for audio in audios],
                prune=False,
            )
        return [
            {"writeback_result": {"deleted": str(audio.audio_file_id)}}
            for audio in audios
        ]


def _assigned_segment(segment: dict, speaker_id: str) -> dict:
    annotations = AudioAnnotations.model_validate(segment["annotations"])
    return {
        **segment,
        "annotations": annotations.model_copy(
            update={
                "speaker_id": speaker_id,
            }
        ).model_dump(mode="json"),
    }
