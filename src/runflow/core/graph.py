from __future__ import annotations

from dataclasses import dataclass

from runflow.core.node import Node
from runflow.core.types import dtype_accepts


@dataclass(frozen=True)
class Endpoint:
    node_id: str
    port: str


@dataclass(frozen=True)
class Edge:
    source: Endpoint
    target: Endpoint


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        return node

    def connect(self, source_node: str, source_port: str, target_node: str, target_port: str) -> None:
        if source_node not in self.nodes:
            raise KeyError(f"Unknown source node: {source_node}")
        if target_node not in self.nodes:
            raise KeyError(f"Unknown target node: {target_node}")

        source_ports = self.nodes[source_node].OUTPUTS
        target_ports = self.nodes[target_node].INPUTS

        if source_port not in source_ports:
            raise KeyError(f"Node {source_node} has no output port {source_port}")
        if target_port not in target_ports:
            raise KeyError(f"Node {target_node} has no input port {target_port}")

        src = source_ports[source_port]
        dst = target_ports[target_port]
        if not dtype_accepts(dst.dtype, src.dtype):
            raise TypeError(
                f"Cannot connect {source_node}.{source_port} ({src.dtype.name}) "
                f"to {target_node}.{target_port} ({dst.dtype.name})"
            )

        self.edges.append(Edge(Endpoint(source_node, source_port), Endpoint(target_node, target_port)))

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target.node_id == node_id]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source.node_id == node_id]

    def source_nodes(self) -> list[Node]:
        targets = {edge.target.node_id for edge in self.edges}
        return [node for node_id, node in self.nodes.items() if node_id not in targets]

    def input_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.IS_INPUT]

    def sink_nodes(self) -> list[Node]:
        sources = {edge.source.node_id for edge in self.edges}
        return [node for node_id, node in self.nodes.items() if node_id not in sources]
