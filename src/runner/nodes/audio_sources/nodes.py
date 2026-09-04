from __future__ import annotations

from collections import deque
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, stable_id
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioFileReference


FETCH_BATCH_SIZE = 1_024


class AudioSourceSettings(StrictSettings):
    source: Literal["selected", "dataset", "all"] = "all"
    audio_file_ids: list[UUID] = Field(default_factory=list)
    dataset_id: UUID | None = None
    include_virtual: bool = False
    limit: int | None = Field(default=None, ge=1)
    selection: Literal["all", "random"] = "all"
    count: int = Field(default=100, ge=1)


class AudioSourceNode(Node):
    NODE_TYPE = "AudioSource"
    DESCRIPTION = "Stream existing audio references from PostgreSQL in bounded pages without loading audio bytes. Choose selected files, a dataset, or the full library, then emit ALL matching rows or a fixed-size sample from a random position in indexed UUID order."
    CATEGORY = "Inputs"
    SETTINGS = AudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._after_id: UUID | None = uuid4() if self.settings.selection == "random" else None
        self._wrapped = False
        self._scanned = 0
        self._emitted = 0
        self._page: deque[AudioFileReference] = deque()
        self._population_count = self._count_refs()
        self._output_count = (
            min(self.settings.count, self._population_count)
            if self.settings.selection == "random"
            else self._population_count
        )

    def remaining_items(self, context) -> int:
        return self._output_count - self._emitted

    async def execute(self, batch, context):
        assert len(batch) == 1, f"{self.id} requires one source task"
        context.check_cancel()
        refs = self._next_refs(context, self.runtime.queue_max_size)
        self._emitted += len(refs)
        return [{"audio": self._audio(ref, context)} for ref in refs]

    def _count_refs(self) -> int:
        total = audio_crud.count_audio_file_references(
            self._dataset_id(),
            self._selected_ids(),
            self.settings.include_virtual,
        )
        return min(total, self.settings.limit) if self.settings.limit is not None else total

    def _audio(self, item: AudioFileReference, context) -> Audio:
        requested = self.settings.count if self.settings.selection == "random" else None
        return Audio(
            audio_file_id=item.id,
            name=item.name,
            data=None,
            sample_rate=int(item.annotations.metadata.get("sample_rate", 0) or 0),
            channels=int(item.annotations.metadata.get("channels", 0) or 0),
            start=0.0,
            end=item.duration,
            annotations=item.annotations.model_copy(update={
                "metadata": {
                    **item.annotations.metadata,
                    "source_batch_id": stable_id("audio_source", context.run_id, self.id),
                    "source_batch_count": self._output_count,
                    "sample_selection": self.settings.selection,
                    "sample_requested_count": requested,
                },
            }),
            language=item.language,
            id=stable_id("audio", item.id, item.name),
            lineage_id=stable_id("audio_ref", item.id),
            byte_length=item.byte_length,
            virtual=item.virtual,
            style_prompt=item.style_prompt,
            voice_prompt=item.voice_prompt,
        )

    def _next_refs(self, context, limit: int) -> list[AudioFileReference]:
        target = min(limit, self._output_count - self._emitted)
        while len(self._page) < target and self._scanned < self._output_count:
            context.check_cancel()
            self._fetch_page()
        return [self._page.popleft() for _ in range(min(target, len(self._page)))]

    def _fetch_page(self) -> None:
        fetch_count = min(FETCH_BATCH_SIZE, self._output_count - self._scanned)
        rows = audio_crud.list_audio_file_references_page(
                self._dataset_id(),
                self._selected_ids(),
                self.settings.include_virtual,
                self._after_id,
                fetch_count,
            )
        if not rows and self.settings.selection == "random" and not self._wrapped:
            self._after_id = None
            self._wrapped = True
            rows = audio_crud.list_audio_file_references_page(
                self._dataset_id(),
                self._selected_ids(),
                self.settings.include_virtual,
                self._after_id,
                fetch_count,
            )
        assert rows, f"AudioSource expected {self._output_count - self._scanned} more database rows"
        self._after_id = rows[-1].id
        self._scanned += len(rows)
        self._page.extend(rows)

    def _dataset_id(self) -> UUID | None:
        if self.settings.source != "dataset":
            return None
        if self.settings.dataset_id is None:
            raise ValueError("AudioSource dataset mode requires dataset_id")
        return self.settings.dataset_id

    def _selected_ids(self) -> list[UUID] | None:
        return self.settings.audio_file_ids if self.settings.source == "selected" else None
