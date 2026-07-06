from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

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


class GraphNodeRequest(BaseModel):
    id: str
    type: str
    x: float = 0
    y: float = 0
    params: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeRequest(BaseModel):
    source_node: str
    source_port: str
    target_node: str
    target_port: str


class InlineGraphRunRequest(BaseModel):
    run_id: str | None = None
    runner_id: str | None = None
    nodes: list[GraphNodeRequest]
    edges: list[GraphEdgeRequest] = Field(default_factory=list)
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


class NodeRunSnapshot(BaseModel):
    node_id: str
    status: str
    loaded: bool
    queue_size: int
    remaining_items: int | None
    running_batches: int
    latest_batch_index: int | None
    latest_message: str
    error: str | None
    counters: dict[str, int]


class RunSnapshot(BaseModel):
    run_id: str
    total_event_count: int
    error_count: int
    event_counts: dict[str, int]
    nodes: list[NodeRunSnapshot]


class StopRunCommand(BaseModel):
    command: Literal["stop"] = "stop"
    run_id: str


class NodeLifecycleCommand(BaseModel):
    command: Literal["load_node", "unload_node"]
    run_id: str
    node_id: str


class NodeLogRequestCommand(BaseModel):
    command: Literal["read_node_log"] = "read_node_log"
    request_id: str
    run_id: str
    node_id: str
    work_dir: Path | None = None


class NodeLogResponseMessage(BaseModel):
    request_id: str
    run_id: str
    node_id: str
    content: str
    truncated: bool
    error: str | None = None


class StartGraphRunCommand(BaseModel):
    command: Literal["start_graph"] = "start_graph"
    request: InlineGraphRunRequest


class RunnerEventMessage(BaseModel):
    event: RunEventResponse


class RunnerHeartbeatMessage(BaseModel):
    runner_id: str
    hostname: str
    process_id: int
    port: int
    gpu_index: int | None
    active_run_ids: list[str]
    created_at: datetime
