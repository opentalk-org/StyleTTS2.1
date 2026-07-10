from collections.abc import Awaitable, Callable
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.task import Packet, Task, lineage_from_value, metadata_from_value
from runflow.runtime.join_builder import NodeJoinBuffers
from runflow.runtime.output_values import output_values
from runflow.runtime.routing import add_to_join_buffer, can_create_single_input_task
from runflow.runtime.scheduler_events import SchedulerEventEmitter


class OutputRouter:
    def __init__(
        self,
        graph: Graph,
        context: ExecutionContext,
        events: SchedulerEventEmitter,
        join_buffers: dict[str, NodeJoinBuffers],
        enqueue: Callable[[str, Task], Awaitable[None]],
    ):
        self.graph = graph
        self.context = context
        self.events = events
        self.join_buffers = join_buffers
        self.enqueue = enqueue

    async def route(self, node: Node, batch: list[Task], outputs: list[dict[str, Any]], batch_index: int) -> None:
        self.context.check_cancel()
        if len(outputs) == len(batch):
            task_for_output = batch
        elif len(batch) == 1:
            task_for_output = [batch[0] for _ in outputs]
        else:
            raise ValueError(f"{node.id} returned {len(outputs)} output item(s) for batch size {len(batch)}")

        for task, output_dict in zip(task_for_output, outputs):
            for port_name, value in output_dict.items():
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
                        lineage_id=lineage_from_value(item, inherited=task.lineage_id),
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
            await self.events.packet_delivered(packet, target_node, target_port)

            if can_create_single_input_task(target_node):
                await self.enqueue(
                    target_node.id,
                    Task(
                        node_id=target_node.id,
                        inputs={target_port: packet.value},
                        input_packets={target_port: packet},
                        lineage_id=packet.lineage_id,
                        metadata=packet.metadata,
                    ),
                )
                continue

            tasks = add_to_join_buffer(target_node, target_port, packet, self.join_buffers)
            if not tasks:
                await self.events.join_waiting(target_node, target_port, packet)
            for task in tasks:
                await self.events.join_ready(task)
                await self.enqueue(target_node.id, task)
