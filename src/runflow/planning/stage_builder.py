from __future__ import annotations

from dataclasses import dataclass

from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.planning.topological_sort import topological_sort


@dataclass
class Stage:
    index: int
    nodes: list[Node]


class StageBuilder:
    """Minimal stage builder.

    This scaffold uses one node per stage. Later you can fuse cheap CPU nodes into
    the same stage or split expensive GPU branches into separate stages.
    """

    def build(self, graph: Graph) -> list[Stage]:
        order = topological_sort(graph)
        return [Stage(index=i, nodes=[graph.nodes[node_id]]) for i, node_id in enumerate(order)]
