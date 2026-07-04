from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.task import Packet, Task


def build_join_tasks(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
) -> list[Task]:
    counts = _single_packet_counts(node, grouped)
    repeat_count = counts[0] if counts else 1
    if any(count != repeat_count for count in counts):
        raise ValueError(f"Cannot join uneven packet counts for node {node.id} lineage {lineage_id}")

    return [_build_task(node, lineage_id, grouped, index) for index in range(repeat_count)]


def pop_ready_join_tasks(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
) -> list[Task]:
    tasks: list[Task] = []
    while _has_join_task(node, grouped):
        tasks.append(_pop_task(node, lineage_id, grouped))
    return tasks


def _single_packet_counts(node: Node, grouped: dict[str, list[Packet]]) -> list[int]:
    return [
        len(grouped[name])
        for name, port in node.INPUTS.items()
        if name in grouped and port.mode != PortMode.LIST
    ]


def _build_task(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
    index: int,
) -> Task:
    inputs: dict[str, object] = {}
    input_packets: dict[str, Packet] = {}
    metadata: dict[str, object] = {}

    for name, port in node.INPUTS.items():
        if name in grouped:
            packets = grouped[name]
            packet = packets[0] if port.mode == PortMode.LIST else packets[index]
            inputs[name] = [p.value for p in packets] if port.mode == PortMode.LIST else packet.value
            input_packets[name] = packet
            metadata.update(packet.metadata)
        elif name in node.params:
            inputs[name] = node.params[name]
        elif port.optional:
            inputs[name] = port.default

    return Task(node.id, inputs, input_packets, lineage_id, metadata=metadata)


def _has_join_task(node: Node, grouped: dict[str, list[Packet]]) -> bool:
    for name, port in node.INPUTS.items():
        if name in node.params or port.optional:
            continue
        if name not in grouped or not grouped[name]:
            return False
    return True


def _pop_task(
    node: Node,
    lineage_id: str,
    grouped: dict[str, list[Packet]],
) -> Task:
    inputs: dict[str, object] = {}
    input_packets: dict[str, Packet] = {}
    metadata: dict[str, object] = {}

    for name, port in node.INPUTS.items():
        if name in grouped:
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
