from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from runflow.core.task import Task


# Unknown payloads need a nonzero weight or they bypass admission control.
DEFAULT_ITEM_BYTES = 1 * 1024 * 1024
DEFAULT_TOTAL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024

_MAX_DEPTH = 3


def _value_bytes(value: Any, depth: int = 0) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Path):
        return 0
    if depth < _MAX_DEPTH:
        if isinstance(value, dict):
            return sum(_value_bytes(item, depth + 1) for item in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            return sum(_value_bytes(item, depth + 1) for item in value)
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("size_bytes", "byte_length", "nbytes"):
            hint = metadata.get(key)
            if isinstance(hint, (int, float)) and hint > 0:
                return int(hint)
    return 0


def estimate_task_bytes(task: Task, default_item_bytes: int = DEFAULT_ITEM_BYTES) -> int:
    if not task.inputs:
        return 0
    total = 0
    for value in task.inputs.values():
        measured = _value_bytes(value)
        total += measured if measured > 0 else default_item_bytes
    return total


class WeightBudget:
    """Bound in-flight payload memory without deadlocking on one oversized item."""

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes <= 0:
            raise ValueError("memory budget must be positive")
        self._budget = int(budget_bytes)
        self._used = 0
        self._condition = asyncio.Condition()

    @property
    def used(self) -> int:
        return self._used

    @property
    def budget_bytes(self) -> int:
        return self._budget

    async def acquire(self, weight: int) -> None:
        if weight < 0:
            raise ValueError("task memory weight must be non-negative")
        need = int(weight)
        async with self._condition:
            await self._condition.wait_for(lambda: self._used == 0 or self._used + need <= self._budget)
            self._used += need

    async def release(self, weight: int) -> None:
        if weight < 0:
            raise ValueError("released memory weight must be non-negative")
        give_back = int(weight)
        if give_back == 0:
            return
        async with self._condition:
            if give_back > self._used:
                raise RuntimeError("released memory exceeds admitted memory")
            self._used -= give_back
            self._condition.notify_all()
