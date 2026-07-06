from __future__ import annotations

import json
from pathlib import Path

from runflow.core.graph import Graph
from runflow.registry.node_registry import NodeRegistry


def load_graph_json(path: Path, registry: NodeRegistry) -> Graph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = Graph()

    for node_data in data["nodes"]:
        node = registry.create(
            node_type=node_data["type"],
            node_id=node_data["id"],
            params=node_data.get("params", {}),
        )
        graph.add_node(node)

    for edge in data["edges"]:
        source_node, source_port = edge["from"]
        target_node, target_port = edge["to"]
        graph.connect(source_node, source_port, target_node, target_port)

    return graph
