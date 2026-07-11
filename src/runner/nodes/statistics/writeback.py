from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JsonPort
from shared.db import database_session
from shared.db.statistics import StatisticsEntryCreate, StatisticsEntryRead
from shared.db.statistics import crud as statistics_crud


class SaveStatisticsEntrySettings(StrictSettings):
    name: str
    dataset_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SaveStatisticsEntryNode(Node):
    NODE_TYPE = "SaveStatisticsEntry"
    DESCRIPTION = "Persist a computed statistics payload to the database as a named statistics entry and pass the saved record on. Give it a name, optionally tag it with a dataset, and attach extra metadata in the settings. Wire the output of the dataset statistics aggregator here so the results are stored and can be browsed later."
    CATEGORY = "Audio"
    SETTINGS = SaveStatisticsEntrySettings
    INPUTS = {"statistics": JsonPort()}
    OUTPUTS = {"statistics_entry": JsonPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1})

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            payload = inputs["statistics"]
            assert isinstance(payload, dict), f"statistics payload must be a dict, got {type(payload).__name__}"
            create_payload = StatisticsEntryCreate(
                name=self.settings.name,
                dataset_id=self.settings.dataset_id,
                payload=payload,
                metadata=self.settings.metadata,
            )
            with database_session() as session:
                entry = statistics_crud.create_statistics_entry(session, create_payload)
                read = StatisticsEntryRead.model_validate(entry).model_dump(mode="json", by_alias=False)
            outputs.append({"statistics_entry": read})
        return outputs
