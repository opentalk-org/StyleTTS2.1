from __future__ import annotations

from runflow.core.graph import Graph
from runflow.registry.node_registry import NodeRegistry
from runner.nodes.registry import register_runner_nodes
from shared.schemas import InlineGraphRunRequest


class GraphNodeBuildError(ValueError):
    def __init__(self, node_id: str, node_type: str, error: Exception) -> None:
        self.node_id = node_id
        self.node_type = node_type
        super().__init__(f"{node_type} node {node_id} is invalid: {type(error).__name__}: {error}")


def build_inline_graph(request: InlineGraphRunRequest) -> Graph:
    registry = register_runner_nodes(NodeRegistry())
    graph = Graph()

    for node_data in request.nodes:
        try:
            node = registry.create(
                node_type=node_data.type,
                node_id=node_data.id,
                params={**node_data.params, "runtime": node_data.runtime},
            )
        except Exception as error:
            raise GraphNodeBuildError(node_data.id, node_data.type, error) from error
        graph.add_node(node)

    for edge in request.edges:
        graph.connect(
            edge.source_node,
            edge.source_port,
            edge.target_node,
            edge.target_port,
        )

    return graph
