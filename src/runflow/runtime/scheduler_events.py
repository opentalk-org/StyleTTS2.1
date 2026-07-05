from __future__ import annotations

import traceback
from collections.abc import Iterable
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.node import Node
from runflow.core.task import Packet, Task


class SchedulerEventEmitter:
    def __init__(self, context: ExecutionContext):
        self.context = context

    async def run_started(self, input_item_count: int, node_ids: Iterable[str]) -> None:
        await self.context.emit_event(
            "run_started",
            message=f"run started with {input_item_count} input item(s)",
            detail={"input_items": input_item_count, "nodes": list(node_ids)},
        )

    async def input_items_discovered(self, node: Node, item_count: int) -> None:
        await self.context.emit_event(
            "input_items_discovered",
            message=f"{node.id} discovered {item_count} item(s)",
            node_id=node.id,
            detail={"item_count": item_count, "window_size": node.runtime.window_size},
        )

    async def input_items_remaining(self, node: Node, item_count: int | None) -> None:
        await self.context.emit_event(
            "input_items_remaining",
            message=f"{node.id} has {item_count} item(s) remaining",
            node_id=node.id,
            detail={"item_count": item_count},
        )

    async def window_started(self, window_index: int, item_count: int, item_counts: dict[str, int]) -> None:
        await self.context.emit_event(
            "window_started",
            message=f"window {window_index} started with {item_count} item(s)",
            detail={"item_count": item_count, "item_counts": item_counts},
        )

    async def window_completed(self, window_index: int, item_count: int, item_counts: dict[str, int]) -> None:
        await self.context.emit_event(
            "window_completed",
            message=f"window {window_index} completed",
            detail={"item_count": item_count, "item_counts": item_counts},
        )

    async def task_enqueued(self, node_id: str, task: Task, queue_size: int) -> None:
        await self.context.emit_event(
            "task_enqueued",
            message=f"queued task for {node_id}",
            node_id=node_id,
            lineage_id=task.lineage_id,
            detail={"queue_size": queue_size, "inputs": list(task.inputs.keys())},
        )

    async def queue_depth(self, node_id: str, queue_size: int) -> None:
        await self.context.emit_event(
            "queue_depth",
            message=f"{node_id} queue has {queue_size} item(s)",
            node_id=node_id,
            detail={"queue_size": queue_size},
        )

    async def batch_started(self, node: Node, worker_index: int, batch_index: int, batch: list[Task]) -> None:
        await self.context.emit_event(
            "batch_started",
            message=f"{node.id} started batch {batch_index} with {len(batch)} item(s)",
            node_id=node.id,
            worker_index=worker_index,
            batch_index=batch_index,
            batch_size=len(batch),
            detail={
                "node_type": node.NODE_TYPE,
                "lineage_ids": [task.lineage_id for task in batch],
                "resources": node.runtime.resource_policy.resources,
            },
        )

    async def batch_completed(self, node: Node, worker_index: int, batch_index: int, batch: list[Task], outputs: list[dict[str, Any]]) -> None:
        await self.context.emit_event(
            "batch_completed",
            message=f"{node.id} completed batch {batch_index}",
            node_id=node.id,
            worker_index=worker_index,
            batch_index=batch_index,
            batch_size=len(batch),
            detail={"output_items": len(outputs)},
        )

    async def node_failed(self, node: Node, worker_index: int, error: Exception) -> None:
        await self.context.emit_event(
            "node_failed",
            message=f"{node.id} failed: {type(error).__name__}: {error}",
            node_id=node.id,
            worker_index=worker_index,
            detail={"traceback": "".join(traceback.format_exception(error))},
        )

    async def node_loaded(self, node: Node) -> None:
        await self.context.emit_event(
            "node_loaded",
            message=f"{node.id} loaded",
            node_id=node.id,
            detail={"node_type": node.NODE_TYPE},
        )

    async def node_unloaded(self, node: Node) -> None:
        await self.context.emit_event(
            "node_unloaded",
            message=f"{node.id} unloaded",
            node_id=node.id,
            detail={"node_type": node.NODE_TYPE},
        )

    async def packet_created(self, packet: Packet, batch_index: int) -> None:
        await self.context.emit_event(
            "packet_created",
            message=f"{packet.node_id}.{packet.port} produced {packet.dtype}",
            node_id=packet.node_id,
            port=packet.port,
            batch_index=batch_index,
            lineage_id=packet.lineage_id,
            detail=self.packet_detail(packet),
        )

    async def packet_delivered(self, packet: Packet, target_node: Node, target_port: str) -> None:
        await self.context.emit_event(
            "packet_delivered",
            message=f"{packet.node_id}.{packet.port} -> {target_node.id}.{target_port}",
            node_id=packet.node_id,
            port=packet.port,
            target_node_id=target_node.id,
            target_port=target_port,
            lineage_id=packet.lineage_id,
            detail=self.packet_detail(packet),
        )

    async def join_waiting(self, target_node: Node, target_port: str, packet: Packet) -> None:
        await self.context.emit_event(
            "join_waiting",
            message=f"{target_node.id} waiting for required inputs",
            node_id=target_node.id,
            port=target_port,
            lineage_id=packet.lineage_id,
        )

    async def join_ready(self, task: Task) -> None:
        await self.context.emit_event(
            "join_ready",
            message=f"{task.node_id} received required inputs",
            node_id=task.node_id,
            lineage_id=task.lineage_id,
            detail={"inputs": list(task.inputs.keys())},
        )

    async def node_progress(self, node: Node, value: Any, batch_index: int) -> None:
        if isinstance(value, dict):
            message = str(value.get("message", ""))
            detail = dict(value)
        elif isinstance(value, (int, float)):
            message = f"{node.id} progress {value}"
            detail = {"current": value, "total": 100}
        else:
            message = str(value)
            detail = {}
        await self.context.emit_event("node_progress", message=message, node_id=node.id, batch_index=batch_index, detail=detail)

    def packet_detail(self, packet: Packet) -> dict[str, Any]:
        value = packet.value
        return {
            "dtype": packet.dtype,
            "value_type": type(value).__name__,
            "value_id": getattr(value, "id", None),
            "metadata": packet.metadata,
        }
