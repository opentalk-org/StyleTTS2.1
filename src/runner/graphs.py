from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from runflow.core.graph import Graph
from runflow.registry.node_registry import NodeRegistry
from runflow.tmp_nodes.register import register_builtin_nodes

from runner.schemas import RunContextRequest


class GraphNodeRequest(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeRequest(BaseModel):
    source_node: str
    source_port: str
    target_node: str
    target_port: str


class InlineGraphRunRequest(BaseModel):
    run_id: str | None = None
    nodes: list[GraphNodeRequest]
    edges: list[GraphEdgeRequest] = Field(default_factory=list)
    context: RunContextRequest = Field(default_factory=RunContextRequest)


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
