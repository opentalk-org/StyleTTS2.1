from collections.abc import Awaitable, Callable
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.task import Packet, Task, metadata_from_value
from runflow.runtime.join_builder import NodeJoinBuffers
from runflow.runtime.lineage_tracker import LineageTracker
from runflow.runtime.output_values import output_values
from runflow.runtime.routing import add_to_join_buffer, can_create_single_input_task
from runflow.runtime.scheduler_events import SchedulerEventEmitter


INPUT_INDEX_OUTPUT = "__input_index__"


def tasks_for_outputs(
    node_id: str,
    batch: list[Task],
    outputs: list[dict[str, Any]],
) -> list[Task]:
    if not outputs:
        return []
    associations = [INPUT_INDEX_OUTPUT in output for output in outputs]
    if any(associations):
        if not all(associations):
            raise ValueError(f"{node_id} returned partial output-to-input association")
        indices = [int(output[INPUT_INDEX_OUTPUT]) for output in outputs]
        if any(index < 0 or index >= len(batch) for index in indices):
            raise ValueError(f"{node_id} returned invalid output-to-input association")
        return [batch[index] for index in indices]
    if len(outputs) == len(batch):
        return batch
    if len(batch) == 1:
        return [batch[0] for _ in outputs]
    raise ValueError(f"{node_id} returned {len(outputs)} output item(s) for batch size {len(batch)}")


class OutputRouter:
    def __init__(
        self,
        graph: Graph,
        context: ExecutionContext,
        events: SchedulerEventEmitter,
        join_buffers: dict[str, NodeJoinBuffers],
        enqueue: Callable[[str, Task], Awaitable[float]],
        lineages: LineageTracker,
    ):
        self.graph = graph
        self.context = context
        self.events = events
        self.join_buffers = join_buffers
        self.enqueue = enqueue
        self.lineages = lineages

    async def route(self, node: Node, batch: list[Task], outputs: list[dict[str, Any]], batch_index: int) -> None:
        self.context.check_cancel()
        task_for_output = tasks_for_outputs(node.id, batch, outputs)

        for output_index, (task, output_dict) in enumerate(zip(task_for_output, outputs)):
            lineage_id = f"{node.id}:{batch_index}:{output_index}" if node.IS_INPUT else task.lineage_id
            if node.IS_INPUT:
                await self.lineages.open_source(lineage_id, node.id)
            try:
                await self._route_output(node, task, output_dict, batch_index, lineage_id)
            finally:
                if node.IS_INPUT and self.lineages.tracks(lineage_id):
                    await self.lineages.close_source(lineage_id)

    async def _route_output(
        self,
        node: Node,
        task: Task,
        output_dict: dict[str, Any],
        batch_index: int,
        lineage_id: str,
    ) -> None:
        for port_name, value in output_dict.items():
            if port_name == INPUT_INDEX_OUTPUT:
                continue
            if port_name == "__progress__":
                self.context.check_cancel()
                await self.events.node_progress(node, value, batch_index)
                continue
            if port_name not in node.OUTPUTS:
                raise KeyError(f"{node.id} returned undeclared output port: {port_name}")

            port = node.OUTPUTS[port_name]
            for item in output_values(node, port_name, port, value):
                self.context.check_cancel()
                packet = Packet(
                    node_id=node.id,
                    port=port_name,
                    dtype=port.TYPE_NAME,
                    value=item,
                    lineage_id=lineage_id,
                    metadata={**task.metadata, **metadata_from_value(item)},
                )
                await self.events.packet_created(packet, batch_index)
                await self._deliver(packet)

    async def _deliver(self, packet: Packet) -> None:
        self.context.check_cancel()
        for edge in self.graph.outgoing_edges(packet.node_id):
            if edge.source.port != packet.port:
                continue

            target_node = self.graph.nodes[edge.target.node_id]
            target_port = edge.target.port
            if can_create_single_input_task(target_node):
                task = Task(
                    node_id=target_node.id,
                    inputs={target_port: packet.value},
                    input_packets={target_port: packet},
                    lineage_id=packet.lineage_id,
                    metadata=packet.metadata,
                )
                blocked_ms = await self._enqueue_tracked(target_node.id, task)
                await self.events.packet_delivered(packet, target_node, target_port, blocked_ms, False)
                continue

            result = add_to_join_buffer(target_node, target_port, packet, self.join_buffers)
            for lineage_id in result.opened_lineages:
                await self.lineages.open_join(lineage_id)
            blocked_ms = 0.0
            if not result.tasks:
                await self.events.join_waiting(target_node, target_port, packet, len(self.join_buffers[target_node.id].item_groups))
            for task in result.tasks:
                await self.events.join_ready(task)
                blocked_ms += await self._enqueue_tracked(target_node.id, task)
            for lineage_id in result.closed_lineages:
                await self.lineages.close_join(lineage_id)
            await self.events.packet_delivered(packet, target_node, target_port, blocked_ms, not result.tasks)

    async def _enqueue_tracked(self, node_id: str, task: Task) -> float:
        await self.lineages.add_task(task.lineage_id)
        try:
            return await self.enqueue(node_id, task)
        except BaseException:
            await self.lineages.finish_task(task.lineage_id)
            raise
