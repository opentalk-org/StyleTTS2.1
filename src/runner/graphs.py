from __future__ import annotations

from runflow.core.graph import Graph
from runflow.registry.node_registry import NodeRegistry
from runflow.tmp_nodes.register import register_builtin_nodes
from shared.schemas import InlineGraphRunRequest


def build_inline_graph(request: InlineGraphRunRequest) -> Graph:
    registry = register_builtin_nodes(NodeRegistry())
    graph = Graph()

    for node_data in request.nodes:
        node = registry.create(
            node_type=node_data.type,
            node_id=node_data.id,
            params={**node_data.params, "runtime": node_data.runtime},
        )
        graph.add_node(node)

    for edge in request.edges:
        graph.connect(
            edge.source_node,
            edge.source_port,
            edge.target_node,
            edge.target_port,
        )

    return graph
