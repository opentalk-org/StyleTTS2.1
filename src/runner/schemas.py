from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from runflow.core.config import RuntimeConfig


class RunState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunContextRequest(BaseModel):
    work_dir: Path = Path("work")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    device: str = "cuda"
    config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    input_items: list[Any] = Field(default_factory=list)


class RunStartRequest(BaseModel):
    workflow_path: Path
    run_id: str | None = None
    context: RunContextRequest = Field(default_factory=RunContextRequest)


class RunStatus(BaseModel):
    run_id: str
    state: RunState
    workflow_path: Path
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    event_count: int = 0


class RunnerStatus(BaseModel):
    total_runs: int
    active_runs: int
    runs: list[RunStatus]


class RunEventResponse(BaseModel):
    sequence: int
    kind: str
    run_id: str
    created_at: datetime
    message: str
    node_id: str | None
    port: str | None
    target_node_id: str | None
    target_port: str | None
    window_index: int | None
    worker_index: int | None
    batch_index: int | None
    batch_size: int | None
    lineage_id: str | None
    detail: dict[str, Any]
