from __future__ import annotations

import asyncio
from time import perf_counter

from runflow.core.context import ExecutionContext
from runflow.core.task import Task
from runflow.runtime.memory_budget import WeightBudget, estimate_task_bytes


class TaskAdmission:
    """Own queue admission and the activity signals used by batch collection."""

    def __init__(
        self,
        context: ExecutionContext,
        condition: asyncio.Condition,
        queues: dict[str, asyncio.Queue[Task]],
        budgets: dict[str, WeightBudget],
    ) -> None:
        self.context = context
        self.condition = condition
        self.queues = queues
        self.budgets = budgets
        self.active_total = 0
        self.active_by_node = {node_id: 0 for node_id in queues}
        self.blocked_by_node = {node_id: 0 for node_id in queues}
        self.task_weights: dict[int, int] = {}
        self.enqueued_at: dict[int, float] = {}

    async def enqueue(self, node_id: str, task: Task) -> float:
        self.context.check_cancel()
        async with self.condition:
            self.active_total += 1
            self.active_by_node[node_id] += 1
        self.enqueued_at[id(task)] = perf_counter()
        weight = estimate_task_bytes(task)
        self.task_weights[id(task)] = weight
        blocked_at = perf_counter()
        budget = self.budgets[node_id]
        admission_blocked = (
            budget.used > 0 and budget.used + weight > budget.budget_bytes
        )
        if admission_blocked:
            async with self.condition:
                self.blocked_by_node[node_id] += 1
                self.condition.notify_all()
        try:
            await budget.acquire(weight)
        except BaseException:
            await self._discard(node_id, task)
            raise
        finally:
            if admission_blocked:
                async with self.condition:
                    self.blocked_by_node[node_id] -= 1
                    self.condition.notify_all()
        count_blocked = self.queues[node_id].full()
        if count_blocked:
            async with self.condition:
                self.blocked_by_node[node_id] += 1
                self.condition.notify_all()
        try:
            await self.queues[node_id].put(task)
        except BaseException:
            await budget.release(self.task_weights[id(task)])
            await self._discard(node_id, task)
            raise
        finally:
            if count_blocked:
                async with self.condition:
                    self.blocked_by_node[node_id] -= 1
                    self.condition.notify_all()
        async with self.condition:
            self.condition.notify_all()
        return (perf_counter() - blocked_at) * 1000

    async def complete(self, task: Task) -> None:
        node_id = task.node_id
        self.enqueued_at.pop(id(task))
        await self.budgets[node_id].release(self.task_weights.pop(id(task)))
        self.queues[node_id].task_done()
        async with self.condition:
            self.active_total -= 1
            self.active_by_node[node_id] -= 1
            self.condition.notify_all()

    async def _discard(self, node_id: str, task: Task) -> None:
        self.enqueued_at.pop(id(task), None)
        self.task_weights.pop(id(task), None)
        async with self.condition:
            self.active_total -= 1
            self.active_by_node[node_id] -= 1
            self.condition.notify_all()
