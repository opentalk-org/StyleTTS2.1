from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from runflow.core.task import Task


# Fallback size for a task input we can't measure (no nbytes / not bytes / lazy Path).
# Sized for the common "many ~1 MB files" workload so unknown items are still bounded.
DEFAULT_ITEM_BYTES = 1 * 1024 * 1024
# Fallback when no per-run budget is supplied (e.g. a scheduler driven directly from a
# test). The runner fills RuntimeConfig.memory_budget_mb from detected RAM in practice.
DEFAULT_TOTAL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024

_MAX_DEPTH = 3


def _value_bytes(value: Any, depth: int = 0) -> int:
    """Best-effort resident-memory estimate for one payload value. Returns 0 when
    unknown; the caller substitutes a default so unknowns are never counted as free."""
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    nbytes = getattr(value, "nbytes", None)  # numpy arrays, torch tensors
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    if isinstance(value, str):
        return len(value)  # ~1 byte/char is close enough for the small strings we see
    if isinstance(value, Path):
        return 0  # a path is a reference, not resident bytes
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
    """Estimate a task's resident footprint by summing its inputs. An input we can't
    measure counts as ``default_item_bytes`` rather than zero, so we never undercount
    and risk OOM. Seed tasks (no inputs) cost ~nothing and are always admitted."""
    if not task.inputs:
        return 0
    total = 0
    for value in task.inputs.values():
        measured = _value_bytes(value)
        total += measured if measured > 0 else default_item_bytes
    return total


class WeightBudget:
    """Async admission control by bytes, decoupled from the queue so the queue keeps its
    native get/get_nowait/qsize/task_done. A producer acquires its task's weight before
    enqueuing and it is released once the task is consumed, so outstanding weight equals
    the bytes resident in (queue + in-flight batch) for that node.

    Deadlock-safe: an item is admitted whenever nothing is outstanding, so a single item
    larger than the whole budget still passes (serialized) instead of blocking forever."""

    def __init__(self, budget_bytes: int) -> None:
        self._budget = max(1, int(budget_bytes))
        self._used = 0
        self._condition = asyncio.Condition()

    @property
    def used(self) -> int:
        return self._used

    @property
    def budget_bytes(self) -> int:
        return self._budget

    async def acquire(self, weight: int) -> None:
        need = max(0, int(weight))
        async with self._condition:
            await self._condition.wait_for(lambda: self._used == 0 or self._used + need <= self._budget)
            self._used += need

    async def release(self, weight: int) -> None:
        give_back = max(0, int(weight))
        if give_back == 0:
            return
        async with self._condition:
            self._used = max(0, self._used - give_back)
            self._condition.notify_all()
