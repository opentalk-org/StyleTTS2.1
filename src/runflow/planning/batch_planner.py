from __future__ import annotations

from typing import Any

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
    def build_batches(self, tasks: list[Task], policy: BatchPolicy) -> list[list[Task]]:
        if not tasks:
            return []

        if policy.mode == BatchMode.DISABLED:
            return [[task] for task in tasks]

        max_size = max(1, policy.preferred_size or policy.max_size or 1)
        max_size = min(max_size, max(1, policy.max_size or max_size))

        group_tasks = list(tasks)

        batches: list[list[Task]] = []
        if policy.sort_by:
            group_tasks = sorted(group_tasks, key=lambda t: _read_attr_or_metadata(t, policy.sort_by) or 0)

        for i in range(0, len(group_tasks), max_size):
            batches.append(group_tasks[i : i + max_size])

        return batches
