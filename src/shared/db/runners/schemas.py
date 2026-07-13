from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RunnerCreate(BaseModel):
    name: str
    hostname: str
    port: int
    gpu_index: int | None = None


class RunnerRead(RunnerCreate):
    id: UUID
    process_id: int | None
    active_run_ids: list[str]
    capabilities: dict[str, Any]
    last_seen_at: datetime | None
    model_config = ConfigDict(from_attributes=True)
