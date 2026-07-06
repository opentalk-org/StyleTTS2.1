from dataclasses import dataclass, field

from shared.schemas import NodeRunSnapshot, RunEventResponse, RunSnapshot


ERROR_EVENT_KINDS = {"node_failed", "node_lifecycle_failed", "run_failed"}


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
            if state.status == "idle" and state.queue_size > 0:
                state.status = "queued"
        elif event.kind == "batch_started":
            state.running_batches += 1
            state.status = "running"
        elif event.kind == "batch_completed":
            state.running_batches = max(0, state.running_batches - 1)
            state.status = "idle" if state.running_batches == 0 else "running"
            self._increment_node_counter(state, "batches_completed")
        elif event.kind == "packet_created":
            self._increment_node_counter(state, "packets_created")
            self._increment_node_counter(state, "tasks_completed")
        elif event.kind == "packet_delivered":
            self._increment_node_counter(state, "packets_delivered")
        elif event.kind == "node_failed":
            state.running_batches = 0
            state.status = "failed"
            state.error = event.message
        elif event.kind == "node_lifecycle_failed":
            state.error = event.message

    def _update_target_node(self, event: RunEventResponse, state: NodeState) -> None:
        if event.kind == "packet_delivered":
            self._increment_node_counter(state, "packets_received")

    def _increment_node_counter(self, state: NodeState, name: str) -> None:
        if name not in state.counters:
            state.counters[name] = 0
        state.counters[name] += 1

    def _add_node_counter(self, state: NodeState, name: str, amount: int) -> None:
        if name not in state.counters:
            state.counters[name] = 0
        state.counters[name] += amount
