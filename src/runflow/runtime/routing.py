from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.task import Packet, Task
from runflow.runtime.join_builder import NodeJoinBuffers, pop_ready_join_tasks


def can_create_single_input_task(node: Node) -> bool:
    required = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
    return len(node.INPUTS) == 1 and len(required) == 1


def add_to_join_buffer(
    node: Node,
    input_name: str,
    packet: Packet,
    join_buffers: dict[str, NodeJoinBuffers],
) -> list[Task]:
    state = join_buffers[node.id]
    port = node.INPUTS[input_name]
    if port.join_mode == JoinMode.BROADCAST:
        state.broadcasts[input_name] = packet
        if _has_no_required_item_inputs(node) and _has_required_broadcasts(node, state.broadcasts):
            return [_broadcast_only_task(node, state.broadcasts, packet.lineage_id)]
        tasks = []
        for lineage_id, grouped in list(state.item_groups.items()):
            tasks.extend(pop_ready_join_tasks(node, lineage_id, grouped, state.broadcasts))
            if not any(grouped.values()):
                del state.item_groups[lineage_id]
        if not state.item_groups and not state.broadcasts:
            del join_buffers[node.id]
        return tasks

    grouped = state.item_groups[packet.lineage_id]
    grouped[input_name].append(packet)

    required = [
        name
        for name, candidate in node.INPUTS.items()
        if not candidate.optional and name not in node.params and candidate.join_mode != JoinMode.BROADCAST
    ]
    if not all(name in grouped or name in node.params for name in required):
        return []

    tasks = pop_ready_join_tasks(node, packet.lineage_id, grouped, state.broadcasts)
    if not any(grouped.values()):
        del state.item_groups[packet.lineage_id]
    return tasks


def _has_no_required_item_inputs(node: Node) -> bool:
    for name, port in node.INPUTS.items():
        if name in node.params or port.optional:
            continue
        if port.join_mode != JoinMode.BROADCAST:
            return False
    return True


def _has_required_broadcasts(node: Node, broadcasts: dict[str, Packet]) -> bool:
    for name, port in node.INPUTS.items():
        if name in node.params or port.optional or port.join_mode != JoinMode.BROADCAST:
            continue
        if name not in broadcasts:
            return False
    return True


def _broadcast_only_task(node: Node, broadcasts: dict[str, Packet], lineage_id: str) -> Task:
    inputs = {}
    input_packets = {}
    metadata = {}
    for name, port in node.INPUTS.items():
        if port.join_mode == JoinMode.BROADCAST and name in broadcasts:
            packet = broadcasts[name]
            inputs[name] = packet.value
            input_packets[name] = packet
            metadata.update(packet.metadata)
        elif name in node.params:
            inputs[name] = node.params[name]
        elif port.optional:
            inputs[name] = port.default
    return Task(node.id, inputs, input_packets, lineage_id, metadata=metadata)
