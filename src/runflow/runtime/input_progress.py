from __future__ import annotations

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


def has_remaining_items(counts: dict[str, int]) -> bool:
    return any(count > 0 for count in counts.values())


def ensure_input_progress(before: dict[str, int], after: dict[str, int]) -> None:
    if before == after and has_remaining_items(before):
        raise RuntimeError("Input nodes did not consume any remaining items")


def processed_counts(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {node_id: before[node_id] - after[node_id] for node_id in before}
