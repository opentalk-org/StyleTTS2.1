from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from shared.schemas import (
    BatchPerformanceSnapshot,
    EdgePerformanceSnapshot,
    GraphPerformanceSnapshot,
    NodePerformanceSnapshot,
    RatePoint,
    RunEventResponse,
)


ROLLING_SECONDS = 30
HISTORY_LIMIT = 60
LATENCY_LIMIT = 512
RECENT_BATCH_LIMIT = 30


@dataclass
class CountBucket:
    timestamp: datetime
    count: int = 0


@dataclass
class ActiveBatch:
    started_at: datetime
    queue_wait_ms: float


@dataclass
class NodePerformanceState:
    arrivals: deque[CountBucket] = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    departures: deque[CountBucket] = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    queue_samples: deque[tuple[datetime, int]] = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    queue_size: int = 0
    queue_capacity: int = 0
    batches: int = 0
    arrived_items: int = 0
    departed_items: int = 0
    busy_ms: float = 0
    resource_wait_ms: float = 0
    execute_ms: float = 0
    downstream_blocked_ms: float = 0
    batch_latencies: list[float] = field(default_factory=list)
    recent_batches: deque[BatchPerformanceSnapshot] = field(default_factory=lambda: deque(maxlen=RECENT_BATCH_LIMIT))
    active_batches: dict[int, ActiveBatch] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeKey:
    source_node: str
    source_port: str
    target_node: str
    target_port: str


@dataclass
class EdgePerformanceState:
    deliveries: deque[CountBucket] = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    delivered_items: int = 0
    enqueue_blocked_ms: float = 0
    join_waiting_items: int = 0


class PerformanceState:
    def __init__(self):
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.started_items = 0
        self.completed_items = 0
        self.abandoned_items = 0
        self.completions: deque[CountBucket] = deque(maxlen=HISTORY_LIMIT)
        self.lineage_latencies: list[float] = []
        self.nodes: dict[str, NodePerformanceState] = {}
        self.edges: dict[EdgeKey, EdgePerformanceState] = {}

    def record(self, event: RunEventResponse) -> None:
        if event.node_id is not None:
            self._node(event.node_id)
        if event.target_node_id is not None:
            self._node(event.target_node_id)
        if event.kind == "run_started":
            self.started_at = event.created_at
        elif event.kind in {"run_completed", "run_failed", "run_stopped"}:
            self.finished_at = event.created_at
            for state in self.nodes.values():
                state.active_batches.clear()
        elif event.kind == "lineage_started":
            self.started_items += 1
        elif event.kind == "lineage_completed":
            self.completed_items += 1
            _increment_bucket(self.completions, event.created_at, 1)
            _record_sample(self.lineage_latencies, float(event.detail["elapsed_ms"]))
        elif event.kind == "lineage_abandoned":
            self.abandoned_items += 1
        self._record_node(event)
        self._record_edge(event)

    def snapshot(
        self,
        now: datetime,
    ) -> tuple[GraphPerformanceSnapshot, dict[str, NodePerformanceSnapshot], list[EdgePerformanceSnapshot]]:
        measured_at = self.finished_at or now
        elapsed = _elapsed_seconds(self.started_at, measured_at)
        graph = GraphPerformanceSnapshot(
            started_items=self.started_items,
            completed_items=self.completed_items,
            inflight_items=self.started_items - self.completed_items - self.abandoned_items,
            abandoned_items=self.abandoned_items,
            rolling_throughput=_rolling_rate(self.completions, measured_at, elapsed),
            average_throughput=self.completed_items / elapsed,
            latency_p50_ms=_percentile(self.lineage_latencies, 50),
            latency_p95_ms=_percentile(self.lineage_latencies, 95),
            history=[RatePoint(timestamp=item.timestamp, count=item.count, rate=float(item.count)) for item in self.completions],
        )
        nodes = {node_id: self._node_snapshot(state, measured_at, elapsed) for node_id, state in self.nodes.items()}
        edges = [self._edge_snapshot(key, state, measured_at, elapsed) for key, state in self.edges.items()]
        return graph, nodes, edges

    def _record_node(self, event: RunEventResponse) -> None:
        if event.kind == "task_enqueued" and event.node_id is not None:
            state = self._node(event.node_id)
            state.arrived_items += 1
            _increment_bucket(state.arrivals, event.created_at, 1)
            self._record_queue(state, event)
        elif event.kind == "queue_depth" and event.node_id is not None:
            self._record_queue(self._node(event.node_id), event)
        elif event.kind == "batch_started" and event.node_id is not None:
            state = self._node(event.node_id)
            state.queue_capacity = int(event.detail["queue_capacity"])
            state.active_batches[int(event.batch_index)] = ActiveBatch(event.created_at, float(event.detail["queue_wait_ms"]))
        elif event.kind == "batch_completed" and event.node_id is not None:
            self._record_batch(self._node(event.node_id), event)
        elif event.kind == "packet_delivered" and event.node_id is not None:
            self._node(event.node_id).downstream_blocked_ms += float(event.detail["enqueue_blocked_ms"])
        elif event.kind == "node_failed" and event.node_id is not None:
            self._node(event.node_id).active_batches.clear()

    def _record_edge(self, event: RunEventResponse) -> None:
        if event.kind != "packet_delivered":
            return
        key = EdgeKey(str(event.node_id), str(event.port), str(event.target_node_id), str(event.target_port))
        state = self.edges.setdefault(key, EdgePerformanceState())
        state.delivered_items += 1
        state.enqueue_blocked_ms += float(event.detail["enqueue_blocked_ms"])
        state.join_waiting_items += int(bool(event.detail["join_waiting"]))
        _increment_bucket(state.deliveries, event.created_at, 1)

    def _record_queue(self, state: NodePerformanceState, event: RunEventResponse) -> None:
        state.queue_size = int(event.detail["queue_size"])
        state.queue_samples.append((event.created_at, state.queue_size))

    def _record_batch(self, state: NodePerformanceState, event: RunEventResponse) -> None:
        detail = event.detail
        batch = BatchPerformanceSnapshot(
            batch_index=int(event.batch_index), input_items=int(detail["input_items"]), output_items=int(detail["output_items"]),
            queue_wait_ms=float(detail["queue_wait_ms"]), resource_wait_ms=float(detail["resource_wait_ms"]),
            load_ms=float(detail["load_ms"]), execute_ms=float(detail["execute_ms"]), unload_ms=float(detail["unload_ms"]),
            route_ms=float(detail["route_ms"]), total_ms=float(detail["total_ms"]),
        )
        state.batches += 1
        state.departed_items += batch.input_items
        state.busy_ms += batch.total_ms
        state.resource_wait_ms += batch.resource_wait_ms
        state.execute_ms += batch.execute_ms
        _increment_bucket(state.departures, event.created_at, batch.input_items)
        _record_sample(state.batch_latencies, batch.total_ms)
        state.recent_batches.append(batch)
        del state.active_batches[batch.batch_index]

    def _node(self, node_id: str) -> NodePerformanceState:
        return self.nodes.setdefault(node_id, NodePerformanceState())

    def _node_snapshot(self, state: NodePerformanceState, now: datetime, elapsed: float) -> NodePerformanceSnapshot:
        current_busy_ms = sum(max(0.0, (now - batch.started_at).total_seconds() * 1000) for batch in state.active_batches.values())
        queue_growth = _queue_growth(state.queue_samples, now, elapsed)
        return NodePerformanceSnapshot(
            batches=state.batches, arrived_items=state.arrived_items, departed_items=state.departed_items,
            arrival_rate=_rolling_rate(state.arrivals, now, elapsed), departure_rate=_rolling_rate(state.departures, now, elapsed),
            queue_size=state.queue_size, queue_capacity=state.queue_capacity,
            queue_fill_ratio=min(1.0, state.queue_size / state.queue_capacity) if state.queue_capacity else 0,
            queue_growth_rate=queue_growth, busy_ratio=min(1.0, (state.busy_ms + current_busy_ms) / (elapsed * 1000)),
            resource_wait_ratio=min(1.0, state.resource_wait_ms / state.busy_ms) if state.busy_ms else 0,
            downstream_blocked_ms=state.downstream_blocked_ms,
            batch_p50_ms=_percentile(state.batch_latencies, 50), batch_p95_ms=_percentile(state.batch_latencies, 95),
            service_capacity=state.departed_items * 1000 / state.execute_ms if state.execute_ms else 0,
            current_batch_started_at=min((item.started_at for item in state.active_batches.values()), default=None),
            recent_batches=list(state.recent_batches),
        )

    def _edge_snapshot(self, key: EdgeKey, state: EdgePerformanceState, now: datetime, elapsed: float) -> EdgePerformanceSnapshot:
        return EdgePerformanceSnapshot(
            source_node=key.source_node, source_port=key.source_port, target_node=key.target_node, target_port=key.target_port,
            delivered_items=state.delivered_items, rolling_rate=_rolling_rate(state.deliveries, now, elapsed),
            enqueue_blocked_ms=state.enqueue_blocked_ms, join_waiting_items=state.join_waiting_items,
        )


def _increment_bucket(buckets: deque[CountBucket], timestamp: datetime, amount: int) -> None:
    second = timestamp.astimezone(timezone.utc).replace(microsecond=0)
    if buckets and buckets[-1].timestamp == second:
        buckets[-1].count += amount
        return
    buckets.append(CountBucket(timestamp=second, count=amount))


def _record_sample(samples: list[float], value: float) -> None:
    if len(samples) == LATENCY_LIMIT:
        del samples[0]
    samples.append(value)


def _elapsed_seconds(started_at: datetime | None, now: datetime) -> float:
    return max(0.001, (now - started_at).total_seconds()) if started_at is not None else 0.001


def _rolling_rate(buckets: deque[CountBucket], now: datetime, elapsed: float) -> float:
    cutoff = now - timedelta(seconds=ROLLING_SECONDS)
    count = sum(item.count for item in buckets if item.timestamp >= cutoff)
    return count / min(float(ROLLING_SECONDS), elapsed)


def _queue_growth(samples: deque[tuple[datetime, int]], now: datetime, elapsed: float) -> float:
    recent = [sample for sample in samples if sample[0] >= now - timedelta(seconds=ROLLING_SECONDS)]
    if len(recent) < 2:
        return 0
    window_seconds = min(float(ROLLING_SECONDS), max(1.0, elapsed))
    return (recent[-1][1] - recent[0][1]) / window_seconds


def _percentile(samples: list[float], percentile: int) -> float:
    if not samples:
        return 0
    ordered = sorted(samples)
    index = max(0, (len(ordered) * percentile + 99) // 100 - 1)
    return ordered[index]
