from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.performance_state import PerformanceState
from shared.schemas import NodePerformanceSnapshot, NodeRunSnapshot, RunEventResponse, RunSnapshot


ERROR_EVENT_KINDS = {"node_failed", "node_lifecycle_failed", "run_failed"}
ACTIVE_NODE_STATUSES = {"queued", "running"}


@dataclass
class NodeState:
    node_id: str
    status: str = "idle"
    loaded: bool = False
    queue_size: int = 0
    remaining_items: int | None = None
    running_batches: int = 0
    processing_items: int = 0
    latest_batch_index: int | None = None
    latest_message: str = ""
    error: str | None = None
    counters: dict[str, int] = field(default_factory=dict)

    def to_snapshot(self, performance: NodePerformanceSnapshot) -> NodeRunSnapshot:
        return NodeRunSnapshot(
            node_id=self.node_id,
            status=self.status,
            loaded=self.loaded,
            queue_size=self.queue_size,
            remaining_items=self.remaining_items,
            running_batches=self.running_batches,
            processing_items=self.processing_items,
            latest_batch_index=self.latest_batch_index,
            latest_message=self.latest_message,
            error=self.error,
            counters=dict(self.counters),
            performance=performance,
        )


@dataclass
class RunEventStore:
    total_event_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    node_states: dict[str, NodeState] = field(default_factory=dict)
    errors: list[RunEventResponse] = field(default_factory=list)
    performance: PerformanceState = field(default_factory=PerformanceState)

    def record(self, event: RunEventResponse) -> None:
        self.total_event_count += 1
        self.event_counts[event.kind] = self.event_counts.setdefault(event.kind, 0) + 1
        self.performance.record(event)
        self._update_node_state(event)
        self._update_terminal_run_state(event)
        if event.kind in ERROR_EVENT_KINDS:
            self.errors.append(event)

    def snapshot(self, run_id: str) -> RunSnapshot:
        graph_performance, node_performance, edges = self.performance.snapshot(datetime.now(timezone.utc))
        nodes = []
        all_node_ids = dict.fromkeys([*self.node_states, *node_performance])
        for node_id in all_node_ids:
            state = self._node_state(node_id)
            performance = node_performance[node_id]
            nodes.append(state.to_snapshot(performance))
        return RunSnapshot(
            run_id=run_id,
            total_event_count=self.total_event_count,
            error_count=len(self.errors),
            event_counts=dict(self.event_counts),
            performance=graph_performance,
            nodes=nodes,
            edges=edges,
        )

    def _node_state(self, node_id: str) -> NodeState:
        return self.node_states.setdefault(node_id, NodeState(node_id=node_id))

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
        elif event.kind in {"task_enqueued", "queue_depth"}:
            state.queue_size = int(event.detail["queue_size"])
            if state.status == "idle" and state.queue_size > 0:
                state.status = "queued"
        elif event.kind == "batch_started":
            state.running_batches += 1
            state.processing_items += event.batch_size or 0
            state.status = "running"
        elif event.kind == "batch_completed":
            state.running_batches = max(0, state.running_batches - 1)
            state.processing_items = max(0, state.processing_items - (event.batch_size or 0))
            state.status = "idle" if state.running_batches == 0 else "running"
            self._increment_node_counter(state, "batches_completed")
            self._add_node_counter(state, "tasks_completed", int(event.detail["input_items"]))
        elif event.kind == "packet_created":
            self._increment_node_counter(state, "packets_created")
        elif event.kind == "packet_delivered":
            self._increment_node_counter(state, "packets_delivered")
        elif event.kind == "node_failed":
            state.running_batches = 0
            state.processing_items = 0
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
            self._finish_active_nodes("stopped", "Aborted: the run failed in another node")
        elif event.kind == "run_completed":
            self._finish_active_nodes("idle", event.message)

    def _finish_active_nodes(self, status: str, message: str) -> None:
        for state in self.node_states.values():
            if state.status not in ACTIVE_NODE_STATUSES and state.running_batches == 0:
                continue
            state.running_batches = 0
            state.processing_items = 0
            state.queue_size = 0
            state.status = status
            state.latest_message = message
            if status == "stopped":
                state.loaded = False

    def _increment_node_counter(self, state: NodeState, name: str) -> None:
        state.counters[name] = state.counters.setdefault(name, 0) + 1

    def _add_node_counter(self, state: NodeState, name: str, amount: int) -> None:
        state.counters[name] = state.counters.setdefault(name, 0) + amount
