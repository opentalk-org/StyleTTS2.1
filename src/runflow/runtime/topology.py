from runflow.core.graph import Graph
from runflow.core.node import Node


def topological_nodes(graph: Graph) -> list[Node]:
    pending = list(graph.nodes)
    ordered: list[str] = []
    while pending:
        ready = [
            node_id for node_id in pending
            if all(edge.source.node_id in ordered for edge in graph.incoming_edges(node_id))
        ]
        if not ready:
            raise RuntimeError("Graph has a cycle")
        ordered.extend(ready)
        pending = [node_id for node_id in pending if node_id not in ready]
    return [graph.nodes[node_id] for node_id in ordered]
