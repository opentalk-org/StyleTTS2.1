from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StatisticsEntryCreate(BaseModel):
    name: str
    dataset_id: UUID | None
    payload: dict[str, Any]
    metadata: dict[str, Any]


class StatisticsEntryRead(StatisticsEntryCreate):
    id: UUID
    created_at: datetime
    metadata: dict[str, Any] = Field(alias="metadata_")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class StatisticsEntrySummary(BaseModel):
    id: UUID
    name: str
    dataset_id: UUID | None
    file_count: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
