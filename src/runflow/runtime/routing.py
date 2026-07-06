from __future__ import annotations

from runflow.core.node import Node
from runflow.core.task import Packet, Task
from runflow.runtime.join_builder import pop_ready_join_tasks


def can_create_single_input_task(node: Node) -> bool:
    required = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
    return len(node.INPUTS) == 1 and len(required) == 1


def add_to_join_buffer(
    node: Node,
    input_name: str,
    packet: Packet,
    join_buffers: dict[tuple[str, str], dict[str, list[Packet]]],
) -> list[Task]:
    key = (node.id, packet.lineage_id)
    grouped = join_buffers[key]
    grouped[input_name].append(packet)

    required = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
    if not all(name in grouped or name in node.params for name in required):
        return []

    tasks = pop_ready_join_tasks(node, packet.lineage_id, grouped)
    if not any(grouped.values()):
        del join_buffers[key]
    return tasks
