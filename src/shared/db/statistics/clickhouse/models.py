from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.db.clickhouse.types import utc_datetime


class StatisticsEntryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    dataset_id: UUID | None
    payload: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime

    _timestamps_utc = field_validator("updated_at", "created_at")(utc_datetime)

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.metadata
