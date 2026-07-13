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
    active = node.status in ACTIVE_NODE_STATUSES or node.running_batches > 0
    performance = node.performance.model_copy(update={"current_batch_started_at": None})
    updates = {
        "loaded": False,
        "queue_size": 0,
        "running_batches": 0,
        "performance": performance,
    }
    if active:
        updates |= {"status": "stopped", "latest_message": message}
    return node.model_copy(update=updates)
