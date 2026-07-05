from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from shared.db.runners.schemas import RunnerCreate, RunnerRead


class RunnerStatusRead(RunnerRead):
    online: bool
    stale: bool
    busy: bool
    active_run_ids: list[str]
    process_id: int | None
    last_seen_at: datetime | None


class RunnerPage(BaseModel):
    rows: list[RunnerStatusRead]
    total: int


class RunnerManualRead(BaseModel):
    id: UUID


RunnerRegisterRequest = RunnerCreate
