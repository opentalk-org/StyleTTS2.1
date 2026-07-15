from __future__ import annotations

from typing import Any, Callable

from runflow.core.task import Task
from runflow.policies import BatchMode, BatchPolicy


def _read_attr_or_metadata(task: Task, key: str) -> Any:
    if key in task.metadata:
        return task.metadata[key]

    for value in task.inputs.values():
        metadata = getattr(value, "metadata", None)
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
        if hasattr(value, key):
            return getattr(value, key)

    return None


class BatchPlanner:
    def build_batches(
        self,
        tasks: list[Task],
        policy: BatchPolicy,
        weight_of: Callable[[Task], int] | None = None,
        byte_target: int | None = None,
    ) -> list[list[Task]]:
        """Split tasks into execute() batches.

        The count cap (preferred/max_size) is the compute ceiling — it still governs the
        common small-item case. When ``weight_of``/``byte_target`` are supplied, a batch is
        *also* capped by bytes: batches only ever get smaller, so big payloads shrink the
        batch below the count cap automatically while small-item behaviour is unchanged. A
        single item larger than the target still forms its own batch (never dropped)."""
        if not tasks:
            return []

        if policy.mode == BatchMode.DISABLED:
            return [[task] for task in tasks]

        count_cap = max(1, policy.preferred_size or policy.max_size or 1)
        count_cap = min(count_cap, max(1, policy.max_size or count_cap))

        group_tasks = list(tasks)
        if policy.sort_by:
            group_tasks = sorted(group_tasks, key=lambda t: _read_attr_or_metadata(t, policy.sort_by) or 0)

        if weight_of is None or not byte_target or byte_target <= 0:
            return [group_tasks[i : i + count_cap] for i in range(0, len(group_tasks), count_cap)]

        batches: list[list[Task]] = []
        current: list[Task] = []
        current_bytes = 0
        for task in group_tasks:
            weight = max(0, weight_of(task))
            if current and (len(current) >= count_cap or current_bytes + weight > byte_target):
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(task)
            current_bytes += weight
        if current:
            batches.append(current)
        return batches
