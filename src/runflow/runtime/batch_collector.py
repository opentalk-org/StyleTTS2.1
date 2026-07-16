from __future__ import annotations

import asyncio
from time import perf_counter

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.task import Task
from runflow.planning.batch_planner import BatchPlanner
from runflow.policies import BatchMode
from runflow.runtime.memory_budget import WeightBudget


class BatchCollector:
    """Collect queued tasks without making batching depend on producer timing."""

    def __init__(
        self,
        graph: Graph,
        context: ExecutionContext,
        condition: asyncio.Condition,
        queues: dict[str, asyncio.Queue[Task]],
        budgets: dict[str, WeightBudget],
        task_weights: dict[int, int],
        active_tasks: dict[str, int],
        blocked_admissions: dict[str, int],
    ) -> None:
        self.graph = graph
        self.context = context
        self.condition = condition
        self.queues = queues
        self.budgets = budgets
        self.task_weights = task_weights
        self.active_tasks = active_tasks
        self.blocked_admissions = blocked_admissions
        self.planner = BatchPlanner()

    async def collect(
        self,
        node: Node,
    ) -> tuple[list[list[Task]], list[Task]]:
        policy = node.runtime.batch_policy.to_policy()
        target_size = 1
        if policy.mode != BatchMode.DISABLED:
            target_size = min(policy.max_size, policy.preferred_size)
        await self._wait_for_items(node.id, 1, None)
        deadline = (
            perf_counter() + policy.timeout_ms / 1000.0
            if policy.timeout_ms > 0
            else None
        )
        await self._wait_for_items(node.id, target_size, deadline)

        queue = self.queues[node.id]
        tasks = []
        while len(tasks) < target_size and not queue.empty():
            tasks.append(queue.get_nowait())

        budget = self.budgets[node.id]
        batches = self.planner.build_batches(
            tasks,
            policy,
            weight_of=lambda task: self.task_weights[id(task)],
            byte_target=budget.budget_bytes,
        )
        return batches, tasks

    async def _wait_for_items(
        self,
        node_id: str,
        target_size: int,
        deadline: float | None,
    ) -> None:
        async with self.condition:
            while not self._ready(node_id, target_size, deadline):
                if deadline is None:
                    await self.condition.wait()
                    continue
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    return
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return

    def _ready(self, node_id: str, target_size: int, deadline: float | None) -> bool:
        queue_size = self.queues[node_id].qsize()
        if queue_size >= target_size:
            return True
        return queue_size > 0 and self._must_flush(node_id, deadline)

    def _must_flush(self, node_id: str, deadline: float | None) -> bool:
        if deadline is not None and perf_counter() >= deadline:
            return True
        if self.blocked_admissions[node_id] > 0:
            return True
        return not self._upstream_can_produce(node_id)

    def _upstream_can_produce(self, node_id: str) -> bool:
        pending = [edge.source.node_id for edge in self.graph.incoming_edges(node_id)]
        visited = set()
        while pending:
            upstream_id = pending.pop()
            if upstream_id in visited:
                continue
            visited.add(upstream_id)
            if self.active_tasks[upstream_id] > 0:
                return True
            upstream = self.graph.nodes[upstream_id]
            if upstream.IS_INPUT and upstream.remaining_items(self.context) > 0:
                return True
            pending.extend(
                edge.source.node_id
                for edge in self.graph.incoming_edges(upstream_id)
            )
        return False
