from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StatisticsEntryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    dataset_id: UUID | None
    payload: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
