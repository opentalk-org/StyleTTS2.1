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


class BatchPerformanceSnapshot(BaseModel):
    batch_index: int
    input_items: int
    output_items: int
    queue_wait_ms: float
    resource_wait_ms: float
    load_ms: float
    execute_ms: float
    unload_ms: float
    route_ms: float
    total_ms: float


class RatePoint(BaseModel):
    timestamp: datetime
    count: int
    rate: float


class GraphPerformanceSnapshot(BaseModel):
    started_items: int
    completed_items: int
    inflight_items: int
    abandoned_items: int
    rolling_throughput: float
    average_throughput: float
    latency_p50_ms: float
    latency_p95_ms: float
    history: list[RatePoint]


class NodePerformanceSnapshot(BaseModel):
    batches: int
    arrived_items: int
    departed_items: int
    arrival_rate: float
    departure_rate: float
    queue_size: int
    queue_capacity: int
    queue_fill_ratio: float
    queue_growth_rate: float
    busy_ratio: float
    resource_wait_ratio: float
    downstream_blocked_ms: float
    batch_p50_ms: float
    batch_p95_ms: float
    service_capacity: float
    current_batch_started_at: datetime | None
    recent_batches: list[BatchPerformanceSnapshot]


class EdgePerformanceSnapshot(BaseModel):
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    delivered_items: int
    rolling_rate: float
    enqueue_blocked_ms: float
    join_waiting_items: int


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
    performance: NodePerformanceSnapshot


class RunSnapshot(BaseModel):
    run_id: str
    total_event_count: int
    error_count: int
    event_counts: dict[str, int]
    performance: GraphPerformanceSnapshot
    nodes: list[NodeRunSnapshot]
    edges: list[EdgePerformanceSnapshot]


class NodeLogResponseMessage(BaseModel):
    request_id: str
    run_id: str
    node_id: str
    content: str
    truncated: bool
    error: str | None = None
