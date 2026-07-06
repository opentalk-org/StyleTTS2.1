from runflow.core.context import ExecutionContext
from runflow.core.node import Node


def remaining_counts(input_nodes: list[Node], context: ExecutionContext) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in input_nodes:
        count = node.remaining_items(context)
        if count is None:
            raise RuntimeError(f"Input node {node.id} must return remaining item count")
        counts[node.id] = count
    return counts
