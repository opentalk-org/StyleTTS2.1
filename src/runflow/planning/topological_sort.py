from __future__ import annotations

from collections import defaultdict, deque

from runflow.core.graph import Graph


def topological_sort(graph: Graph) -> list[str]:
    indegree = {node_id: 0 for node_id in graph.nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges:
        outgoing[edge.source.node_id].append(edge.target.node_id)
        indegree[edge.target.node_id] += 1

    queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
    result: list[str] = []

    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(result) != len(graph.nodes):
        raise ValueError("Graph contains a cycle or disconnected dependency issue")

    return result
