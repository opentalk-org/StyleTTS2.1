from __future__ import annotations

from collections import deque
from typing import Iterable

from runflow.core.task import Task


class TaskQueue:
    def __init__(self, tasks: Iterable[Task] = ()): 
        self._items = deque(tasks)

    def push(self, task: Task) -> None:
        self._items.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        self._items.extend(tasks)

    def pop_many(self, max_items: int) -> list[Task]:
        items: list[Task] = []
        for _ in range(min(max_items, len(self._items))):
            items.append(self._items.popleft())
        return items

    def __len__(self) -> int:
        return len(self._items)
