from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobUpsert(BaseModel):
    run_id: str
    name: str
    state: str
    graph_request: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None


class JobRead(JobUpsert):
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class JobPage(BaseModel):
    rows: list[JobRead]
    total: int


class NodeLogUpsert(BaseModel):
    run_id: str
    node_id: str
    content: str
    truncated: bool
    error: str | None = None


class NodeLogRead(NodeLogUpsert):
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)
