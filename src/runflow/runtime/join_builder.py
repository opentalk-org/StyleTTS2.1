from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, PortMode
from runflow.core.task import Packet, Task


@dataclass
class NodeJoinBuffers:
    item_groups: dict[str, dict[str, list[Packet]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    broadcasts: dict[str, Packet] = field(default_factory=dict)


def pop_ready_join_tasks(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
    broadcasts: dict[str, Packet] | None = None,
) -> list[Task]:
    broadcasts = broadcasts or {}
    tasks: list[Task] = []
    while _has_join_task(node, grouped, broadcasts):
        tasks.append(_pop_task(node, lineage_id, grouped, broadcasts))
    return tasks


def _has_join_task(node: Node, grouped: dict[str, list[Packet]], broadcasts: dict[str, Packet]) -> bool:
    for name, port in node.INPUTS.items():
        if name in node.params or port.optional:
            continue
        if port.join_mode == JoinMode.BROADCAST:
            if name not in broadcasts:
                return False
            continue
        if name not in grouped or not grouped[name]:
            return False
    return True


def _pop_task(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
    broadcasts: dict[str, Packet],
) -> Task:
    inputs: dict[str, object] = {}
    input_packets: dict[str, Packet] = {}
    metadata: dict[str, object] = {}

    for name, port in node.INPUTS.items():
        if port.join_mode == JoinMode.BROADCAST and name in broadcasts:
            packet = broadcasts[name]
            inputs[name] = packet.value
            input_packets[name] = packet
            metadata.update(packet.metadata)
        elif name in grouped:
            packets = grouped[name]
            if port.mode == PortMode.LIST:
                inputs[name] = [p.value for p in packets]
                packet = packets[0]
                grouped[name] = []
            else:
                packet = packets.pop(0)
                inputs[name] = packet.value
            input_packets[name] = packet
            metadata.update(packet.metadata)
        elif name in node.params:
            inputs[name] = node.params[name]
        elif port.optional:
            inputs[name] = port.default

    return Task(node.id, inputs, input_packets, lineage_id, metadata=metadata)
