from dataclasses import dataclass, field
from datetime import datetime

from shared.schemas import BatchPerformanceSnapshot, NodePerformanceSnapshot, NodeRunSnapshot, RunEventResponse, RunSnapshot


ERROR_EVENT_KINDS = {"node_failed", "node_lifecycle_failed", "run_failed"}
ACTIVE_NODE_STATUSES = {"queued", "running"}
RECENT_BATCH_LIMIT = 30
LATENCY_SAMPLE_LIMIT = 200


@dataclass
class ActiveBatchPerformance:
    started_at: datetime
    queue_wait_ms: float


@dataclass
class NodePerformanceState:
    batches: int = 0
    items: int = 0
    max_queue_size: int = 0
    total_queue_wait_ms: float = 0
    total_resource_wait_ms: float = 0
    total_load_ms: float = 0
    total_execute_ms: float = 0
    total_unload_ms: float = 0
    total_route_ms: float = 0
    max_batch_ms: float = 0
    latency_samples: list[float] = field(default_factory=list)
    recent_batches: list[BatchPerformanceSnapshot] = field(default_factory=list)
    active_batches: dict[int, ActiveBatchPerformance] = field(default_factory=dict)

    def start_batch(self, event: RunEventResponse) -> None:
        self.active_batches[int(event.batch_index)] = ActiveBatchPerformance(
            started_at=event.created_at,
            queue_wait_ms=float(event.detail["queue_wait_ms"]),
        )

    def record_batch(self, event: RunEventResponse) -> None:
        detail = event.detail
        batch = BatchPerformanceSnapshot(
            batch_index=int(event.batch_index),
            batch_size=int(event.batch_size),
            queue_wait_ms=float(detail["queue_wait_ms"]),
            resource_wait_ms=float(detail["resource_wait_ms"]),
            load_ms=float(detail["load_ms"]),
            execute_ms=float(detail["execute_ms"]),
            unload_ms=float(detail["unload_ms"]),
            route_ms=float(detail["route_ms"]),
            total_ms=float(detail["total_ms"]),
        )
        self.batches += 1
        self.items += batch.batch_size
        self.total_queue_wait_ms += batch.queue_wait_ms
        self.total_resource_wait_ms += batch.resource_wait_ms
        self.total_load_ms += batch.load_ms
        self.total_execute_ms += batch.execute_ms
        self.total_unload_ms += batch.unload_ms
        self.total_route_ms += batch.route_ms
        self.max_batch_ms = max(self.max_batch_ms, batch.total_ms)
        self.latency_samples = [*self.latency_samples[-(LATENCY_SAMPLE_LIMIT - 1):], batch.total_ms]
        self.recent_batches = [*self.recent_batches[-(RECENT_BATCH_LIMIT - 1):], batch]
        self.active_batches.pop(batch.batch_index)

    def to_snapshot(self) -> NodePerformanceSnapshot:
        ordered = sorted(self.latency_samples)
        p95_index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
        p95 = ordered[p95_index] if ordered else 0
        active = min(self.active_batches.values(), key=lambda batch: batch.started_at) if self.active_batches else None
        measured_total = self.total_resource_wait_ms + self.total_load_ms + self.total_execute_ms + self.total_unload_ms + self.total_route_ms
        return NodePerformanceSnapshot(
            batches=self.batches,
            items=self.items,
            max_queue_size=self.max_queue_size,
            total_queue_wait_ms=self.total_queue_wait_ms,
            total_resource_wait_ms=self.total_resource_wait_ms,
            total_load_ms=self.total_load_ms,
            total_execute_ms=self.total_execute_ms,
            total_unload_ms=self.total_unload_ms,
            total_route_ms=self.total_route_ms,
            average_batch_ms=measured_total / self.batches if self.batches else 0,
            p95_batch_ms=p95,
            max_batch_ms=self.max_batch_ms,
            average_batch_size=self.items / self.batches if self.batches else 0,
            items_per_second=self.items * 1000 / self.total_execute_ms if self.total_execute_ms else 0,
            current_batch_started_at=active.started_at if active is not None else None,
            current_queue_wait_ms=active.queue_wait_ms if active is not None else 0,
            recent_batches=list(self.recent_batches),
        )


@dataclass
class NodeState:
    node_id: str
    status: str = "idle"
    loaded: bool = False
    queue_size: int = 0
    remaining_items: int | None = None
    running_batches: int = 0
    latest_batch_index: int | None = None
    latest_message: str = ""
    error: str | None = None
    counters: dict[str, int] = field(default_factory=dict)
    performance: NodePerformanceState = field(default_factory=NodePerformanceState)

    def to_snapshot(self) -> NodeRunSnapshot:
        remaining_items = self.remaining_items
        if "input_items_discovered" in self.counters:
            completed = self.counters["tasks_completed"] if "tasks_completed" in self.counters else 0
            remaining_items = max(0, self.counters["input_items_discovered"] - completed)
        return NodeRunSnapshot(
            node_id=self.node_id,
            status=self.status,
            loaded=self.loaded,
            queue_size=self.queue_size,
            remaining_items=remaining_items,
            running_batches=self.running_batches,
            latest_batch_index=self.latest_batch_index,
            latest_message=self.latest_message,
            error=self.error,
            counters=dict(self.counters),
            performance=self.performance.to_snapshot(),
        )


@dataclass
class RunEventStore:
    total_event_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    node_states: dict[str, NodeState] = field(default_factory=dict)
    errors: list[RunEventResponse] = field(default_factory=list)

    def record(self, event: RunEventResponse) -> None:
        self.total_event_count += 1
        self._increment_event_count(event.kind)
        self._update_node_state(event)
        self._update_terminal_run_state(event)
        if event.kind in ERROR_EVENT_KINDS:
            self.errors.append(event)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return RunSnapshot(
            run_id=run_id,
            total_event_count=self.total_event_count,
            error_count=len(self.errors),
            event_counts=dict(self.event_counts),
            nodes=[state.to_snapshot() for state in self.node_states.values()],
        )

    def _increment_event_count(self, kind: str) -> None:
        if kind not in self.event_counts:
            self.event_counts[kind] = 0
        self.event_counts[kind] += 1

    def _node_state(self, node_id: str) -> NodeState:
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeState(node_id=node_id)
        return self.node_states[node_id]

    def _update_node_state(self, event: RunEventResponse) -> None:
        if event.node_id is not None:
            self._update_primary_node(event, self._node_state(event.node_id))
        if event.target_node_id is not None:
            self._update_target_node(event, self._node_state(event.target_node_id))

    def _update_primary_node(self, event: RunEventResponse, state: NodeState) -> None:
        state.latest_message = event.message
        state.latest_batch_index = event.batch_index

        if event.kind == "node_loaded":
            state.loaded = True
        elif event.kind == "node_unloaded":
            state.loaded = False
        elif event.kind == "input_items_discovered":
            item_count = event.detail["item_count"]
            state.remaining_items = int(item_count) if item_count is not None else None
            self._add_node_counter(state, "input_items_discovered", int(item_count or 0))
        elif event.kind == "input_items_remaining":
            item_count = event.detail["item_count"]
            state.remaining_items = int(item_count) if item_count is not None else None
        elif event.kind == "task_enqueued" or event.kind == "queue_depth":
            state.queue_size = int(event.detail["queue_size"])
            state.performance.max_queue_size = max(state.performance.max_queue_size, state.queue_size)
            if state.status == "idle" and state.queue_size > 0:
                state.status = "queued"
        elif event.kind == "batch_started":
            state.running_batches += 1
            state.status = "running"
            state.performance.start_batch(event)
        elif event.kind == "batch_completed":
            state.running_batches = max(0, state.running_batches - 1)
            state.status = "idle" if state.running_batches == 0 else "running"
            self._increment_node_counter(state, "batches_completed")
            state.performance.record_batch(event)
        elif event.kind == "packet_created":
            self._increment_node_counter(state, "packets_created")
            self._increment_node_counter(state, "tasks_completed")
        elif event.kind == "packet_delivered":
            self._increment_node_counter(state, "packets_delivered")
        elif event.kind == "node_failed":
            state.running_batches = 0
            state.performance.active_batches.clear()
            state.status = "failed"
            state.error = event.message
        elif event.kind == "node_lifecycle_failed":
            state.error = event.message

    def _update_target_node(self, event: RunEventResponse, state: NodeState) -> None:
        if event.kind == "packet_delivered":
            self._increment_node_counter(state, "packets_received")

    def _update_terminal_run_state(self, event: RunEventResponse) -> None:
        if event.kind == "run_stopped":
            self._finish_active_nodes("stopped", event.message)
        elif event.kind == "run_failed":
            # Only the node that raised gets `failed` + its own error via the earlier
            # `node_failed` event. Other nodes that were merely in flight when the run
            # aborted are marked `stopped` here -- stamping every one of them with this
            # failure's message is what made the same error show up across many nodes.
            self._finish_active_nodes("stopped", "Aborted: the run failed in another node")
        elif event.kind == "run_completed":
            self._finish_active_nodes("idle", event.message)

    def _finish_active_nodes(self, status: str, message: str) -> None:
        for state in self.node_states.values():
            if state.status not in ACTIVE_NODE_STATUSES and state.running_batches == 0:
                continue
            state.running_batches = 0
            state.performance.active_batches.clear()
            state.queue_size = 0
            state.status = status
            state.latest_message = message
            if status == "failed" and state.error is None:
                state.error = message

    def _increment_node_counter(self, state: NodeState, name: str) -> None:
        if name not in state.counters:
            state.counters[name] = 0
        state.counters[name] += 1

    def _add_node_counter(self, state: NodeState, name: str, amount: int) -> None:
        if name not in state.counters:
            state.counters[name] = 0
        state.counters[name] += amount
