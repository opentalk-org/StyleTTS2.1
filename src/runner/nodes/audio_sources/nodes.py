from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AUDIO_REF
from runner.nodes.models import AudioRecordRef, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.common import one
from shared.db.datasets.models import Dataset


class SelectedAudioSourceSettings(StrictSettings):
    audio_file_ids: list[UUID]
    include_virtual: bool = False


class DatasetAudioSourceSettings(StrictSettings):
    dataset_id: UUID
    include_virtual: bool = False


class AllAudioSourceSettings(StrictSettings):
    include_virtual: bool = False
    limit: int | None = Field(default=None, ge=1)


class AudioSourceNode(Node):
    CATEGORY = "Audio / Sources"
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF, mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._items = self._load_refs()
        self._cursor = 0

    def remaining_items(self, context) -> int:
        return len(self._items) - self._cursor

    async def execute(self, batch, context):
        end = self._cursor + self.runtime.queue_max_size
        items = self._items[self._cursor:end]
        self._cursor += len(items)
        return [{"audio_ref": item} for item in items]

    def _load_refs(self) -> list[AudioRecordRef]:
        raise NotImplementedError

    def _ref(self, item: AudioFile) -> AudioRecordRef:
        return AudioRecordRef(
            audio_file_id=item.id,
            name=item.name,
            duration=item.duration,
            byte_length=item.byte_length,
            virtual=item.virtual,
            metadata=item.metadata_,
        )

    def _visible(self, item: AudioFile) -> bool:
        return self.settings.include_virtual or not item.virtual

    def _with_source_batch(self, refs: list[AudioRecordRef]) -> list[AudioRecordRef]:
        batch_id = stable_id("audio_source", self.id, *(ref.audio_file_id for ref in refs))
        return [
            replace(
                ref,
                metadata={
                    **ref.metadata,
                    "source_batch_id": batch_id,
                    "source_batch_count": len(refs),
                },
            )
            for ref in refs
        ]


class SelectedAudioSourceNode(AudioSourceNode):
    NODE_TYPE = "SelectedAudioSource"
    SETTINGS = SelectedAudioSourceSettings

    def _load_refs(self) -> list[AudioRecordRef]:
        with database_session() as session:
            items = [audio_crud.get_audio_file(session, audio_file_id) for audio_file_id in self.settings.audio_file_ids]
            return self._with_source_batch([self._ref(item) for item in items if self._visible(item)])


class DatasetAudioSourceNode(AudioSourceNode):
    NODE_TYPE = "DatasetAudioSource"
    SETTINGS = DatasetAudioSourceSettings

    def _load_refs(self) -> list[AudioRecordRef]:
        with database_session() as session:
            dataset = one(session, Dataset, self.settings.dataset_id)
            return self._with_source_batch([self._ref(item) for item in dataset.audio_files if self._visible(item)])


class AllAudioSourceNode(AudioSourceNode):
    NODE_TYPE = "AllAudioSource"
    SETTINGS = AllAudioSourceSettings

    def _load_refs(self) -> list[AudioRecordRef]:
        with database_session() as session:
            refs = [self._ref(item) for item in audio_crud.list_audio_files(session) if self._visible(item)]
        if self.settings.limit is None:
            return self._with_source_batch(refs)
        return self._with_source_batch(refs[: self.settings.limit])
