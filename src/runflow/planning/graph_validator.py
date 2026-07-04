from __future__ import annotations

from runflow.core.graph import Graph


class GraphValidator:
    def validate(self, graph: Graph) -> None:
        for node in graph.nodes.values():
            incoming_ports = {edge.target.port for edge in graph.incoming_edges(node.id)}
            missing = []
            for name, port in node.INPUTS.items():
                if port.optional:
                    continue
                if name in incoming_ports:
                    continue
                if name in node.params:
                    continue
                missing.append(name)
            if missing:
                raise ValueError(f"Node {node.id} is missing required inputs: {missing}")
