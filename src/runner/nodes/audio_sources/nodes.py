from __future__ import annotations

from dataclasses import replace
from typing import Literal
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.common import one
from shared.db.datasets.models import Dataset


class AudioSourceSettings(StrictSettings):
    source: Literal["selected", "dataset", "all"] = "all"
    audio_file_ids: list[UUID] = Field(default_factory=list)
    dataset_id: UUID | None = None
    include_virtual: bool = False
    limit: int | None = Field(default=None, ge=1)


class AudioSourceNode(Node):
    NODE_TYPE = "AudioSource"
    CATEGORY = "Inputs"
    SETTINGS = AudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
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
        return [{"audio": item} for item in items]

    def _load_refs(self) -> list[Audio]:
        with database_session() as session:
            if self.settings.source == "selected":
                items = [audio_crud.get_audio_file(session, audio_file_id) for audio_file_id in self.settings.audio_file_ids]
            elif self.settings.source == "dataset":
                if self.settings.dataset_id is None:
                    raise ValueError("AudioSource dataset mode requires dataset_id")
                dataset = one(session, Dataset, self.settings.dataset_id)
                items = list(dataset.audio_files)
            else:
                items = audio_crud.list_audio_files(session)
            refs = [self._ref(item) for item in items if self._visible(item)]
        if self.settings.limit is not None:
            refs = refs[: self.settings.limit]
        return self._with_source_batch(refs)

    def _ref(self, item: AudioFile) -> Audio:
        return Audio(
            audio_file_id=item.id,
            name=item.name,
            data=None,
            sample_rate=int(item.metadata_.get("sample_rate", 0) or 0),
            channels=int(item.metadata_.get("channels", 0) or 0),
            start=0.0,
            end=item.duration,
            confidence=1.0,
            id=stable_id("audio", item.id, item.name),
            lineage_id=stable_id("audio_ref", item.id),
            metadata=item.metadata_,
            byte_length=item.byte_length,
            virtual=item.virtual,
        )

    def _visible(self, item: AudioFile) -> bool:
        return self.settings.include_virtual or not item.virtual

    def _with_source_batch(self, refs: list[Audio]) -> list[Audio]:
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
