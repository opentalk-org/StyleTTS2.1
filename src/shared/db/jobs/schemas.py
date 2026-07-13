from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobUpsert(BaseModel):
    run_id: str
    name: str
    state: str
    desired_state: str = "running"
    target_runner_id: str | None = None
    claimed_runner_id: str | None = None
    lease_expires_at: datetime | None = None
    graph_request: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    snapshot: dict[str, Any] | None = None


class JobRead(JobUpsert):
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class JobSummary(BaseModel):
    """Lightweight row for the jobs list: omits the heavy per-row `graph_request` and
    `snapshot` (loaded on demand via `/jobs/{id}/graph` and `/runs/{id}/snapshot`)."""

    run_id: str
    name: str
    state: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class JobPage(BaseModel):
    rows: list[JobSummary]
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


class ClaimedJob(BaseModel):
    run_id: str
    graph_request: dict[str, Any]


class JobStateReplacement(BaseModel):
    run_id: str
    state: str
    snapshot: dict[str, Any]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    release_claim: bool = False


class NodeStateReplacement(BaseModel):
    run_id: str
    node_id: str
    desired_loaded: bool
    observed_loaded: bool | None = None
    error: str | None = None


class RunnerStateFlush(BaseModel):
    runner_id: str
    hostname: str
    port: int
    gpu_index: int | None
    process_id: int
    active_run_ids: list[str]
    capabilities: dict[str, Any]
    jobs: list[JobStateReplacement] = Field(default_factory=list)
    node_states: list[NodeStateReplacement] = Field(default_factory=list)
    logs: list[NodeLogUpsert] = Field(default_factory=list)
