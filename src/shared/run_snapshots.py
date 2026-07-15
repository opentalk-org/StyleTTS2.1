from shared.schemas import NodeRunSnapshot, RunSnapshot


ACTIVE_NODE_STATUSES = {"queued", "running"}


def stopped_run_snapshot(snapshot: RunSnapshot, message: str) -> RunSnapshot:
    event_counts = dict(snapshot.event_counts)
    if "run_stopped" not in event_counts:
        event_counts["run_stopped"] = 0
    event_counts["run_stopped"] += 1
    performance = snapshot.performance.model_copy(
        update={
            "abandoned_items": snapshot.performance.abandoned_items + snapshot.performance.inflight_items,
            "inflight_items": 0,
        }
    )
    return snapshot.model_copy(
        update={
            "total_event_count": snapshot.total_event_count + 1,
            "event_counts": event_counts,
            "performance": performance,
            "nodes": [_stopped_node(node, message) for node in snapshot.nodes],
        }
    )


def _stopped_node(node: NodeRunSnapshot, message: str) -> NodeRunSnapshot:
    return _terminal_node(node, "stopped", message)


def failed_run_snapshot(snapshot: RunSnapshot, message: str) -> RunSnapshot:
    """Turn a live snapshot into a failed one — used when a runner dies mid-run (e.g. OOM)
    and never gets to write its own terminal state. Active nodes are marked failed with
    ``message`` so the graph shows the error instead of nodes stuck 'running'."""
    performance = snapshot.performance.model_copy(
        update={
            "abandoned_items": snapshot.performance.abandoned_items + snapshot.performance.inflight_items,
            "inflight_items": 0,
        }
    )
    return snapshot.model_copy(
        update={
            "performance": performance,
            "nodes": [_failed_node(node, message) for node in snapshot.nodes],
        }
    )


def _failed_node(node: NodeRunSnapshot, message: str) -> NodeRunSnapshot:
    return _terminal_node(node, "failed", message)


def _terminal_node(node: NodeRunSnapshot, status: str, message: str) -> NodeRunSnapshot:
    active = node.status in ACTIVE_NODE_STATUSES or node.running_batches > 0
    performance = node.performance.model_copy(update={"current_batch_started_at": None})
    updates = {
        "loaded": False,
        "queue_size": 0,
        "running_batches": 0,
        "processing_items": 0,
        "performance": performance,
    }
    if active:
        updates |= {"status": status, "latest_message": message}
        if status == "failed" and node.error is None:
            updates |= {"error": message}
    return node.model_copy(update=updates)
