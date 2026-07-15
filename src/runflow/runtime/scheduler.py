import asyncio
import logging
from collections import defaultdict
from time import perf_counter
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.task import Task
from runflow.planning.batch_planner import BatchPlanner
from runflow.planning.graph_validator import GraphValidator
from runflow.runtime.cancellation import cancellation_scope
from runflow.runtime.input_progress import remaining_counts
from runflow.runtime.lineage_tracker import LineageTracker
from runflow.runtime.log_capture import route_output_to_logger
from runflow.runtime.memory_budget import (
    DEFAULT_TOTAL_BUDGET_BYTES,
    WeightBudget,
    estimate_task_bytes,
)
from runflow.runtime.node_manager import NodeManager
from runflow.runtime.output_router import OutputRouter
from runflow.runtime.resource_pool import ResourcePool
from runflow.runtime.join_builder import NodeJoinBuffers
from runflow.runtime.scheduler_events import SchedulerEventEmitter
from runflow.runtime.topology import topological_nodes

class WindowedScheduler:
    def __init__(self, graph: Graph, context: ExecutionContext):
        self.graph = graph
        self.context = context
        self.logger = logging.getLogger(f"runflow.scheduler.{context.run_id}")
        self.validator = GraphValidator()
        self.events = SchedulerEventEmitter(context)
        self.lineages = LineageTracker(self.events)
        self.batch_planner = BatchPlanner()
        self.node_manager = NodeManager(context)

        self.resource_pool = ResourcePool(limits=dict(context.config.resources))

        self.queues: dict[str, asyncio.Queue[Task]] = {}
        self.budgets: dict[str, WeightBudget] = {}
        self._task_weight: dict[int, int] = {}
        self.join_buffers: dict[str, NodeJoinBuffers] = defaultdict(NodeJoinBuffers)
        self.output_router = OutputRouter(graph, context, self.events, self.join_buffers, self._enqueue, self.lineages)
        self._active_tasks = 0
        self._active_condition: asyncio.Condition | None = None
        self._workers: list[asyncio.Task] = []
        self._worker_errors: list[BaseException] = []
        self._batch_sequence = 0
        self._enqueued_at: dict[int, float] = {}

    def run(self) -> None:
        asyncio.run(self.arun())

    def cancel(self) -> None:
        self.context.cancel()

    async def load_node(self, node_id: str) -> None:
        self.context.check_cancel()
        node = self.graph.nodes[node_id]
        was_loaded = node.id in self.node_manager.loaded
        await self.node_manager.ensure_loaded(node)
        if not was_loaded and node.id in self.node_manager.loaded:
            await self.events.node_loaded(node)

    async def unload_node(self, node_id: str) -> None:
        node = self.graph.nodes[node_id]
        was_loaded = node.id in self.node_manager.loaded
        await self.node_manager.unload(node)
        if was_loaded and node.id not in self.node_manager.loaded:
            await self.events.node_unloaded(node)

    async def arun(self) -> None:
        self.logger.info("run starting nodes=%s", len(self.graph.nodes))
        self.validator.validate(self.graph)
        input_nodes = self.graph.input_nodes()
        counts = remaining_counts(input_nodes, self.context)
        total_items = sum(counts.values())
        await self.events.run_started(total_items, self.graph.nodes.keys())
        for node in input_nodes:
            await self.events.input_items_discovered(node, counts[node.id])
            await self.events.input_items_remaining(node, counts[node.id])

        try:
            await self._run_until_idle(input_nodes, counts)
            self.context.check_cancel()
            await self.context.emit_event("run_completed", message="run completed")
            self.logger.info("run completed")
        except BaseException as error:
            await self.lineages.abandon_all(str(error) or type(error).__name__)
            raise
        finally:
            await self._stop_workers()
            for node_id in list(self.node_manager.loaded):
                await self.unload_node(node_id)

    async def _run_until_idle(self, input_nodes: list[Node], remaining_counts: dict[str, int]) -> None:
        self.queues = {
            node_id: asyncio.Queue(maxsize=node.runtime.queue_max_size)
            for node_id, node in self.graph.nodes.items()
        }
        # Spread the total in-flight memory budget across node queues so resident payload
        # bytes are bounded regardless of item size (64x1 MB and 1x500 MB both respect it),
        # replacing per-node queue_max_size as the memory guard. queue_max_size stays as a
        # secondary count cap. Per-queue budget can be smaller than one big item; that's
        # fine — WeightBudget still admits one item at a time, so it never deadlocks.
        per_queue_budget = max(1, self._total_memory_budget_bytes() // max(1, len(self.queues)))
        self.budgets = {node_id: WeightBudget(per_queue_budget) for node_id in self.queues}
        self._task_weight = {}
        self.join_buffers.clear()
        self._active_tasks = 0
        self._active_condition = asyncio.Condition()
        self._workers = []
        self._worker_errors = []
        self._batch_sequence = 0
        self._enqueued_at = {}

        for priority, node in enumerate(topological_nodes(self.graph)):
            self._workers.append(asyncio.create_task(self._worker(node, 0, priority), name=f"{node.id}:0"))

        for source in input_nodes:
            if remaining_counts[source.id] <= 0:
                continue
            await self._enqueue_input(source)

        await self._wait_until_idle()
        await self._stop_workers()

    async def _stop_workers(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    def _raise_worker_errors(self) -> None:
        errors = list(self._worker_errors)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Scheduler worker failures", errors)

    async def _enqueue(self, node_id: str, task: Task) -> float:
        self.context.check_cancel()
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._active_tasks += 1
        self._enqueued_at[id(task)] = perf_counter()
        weight = estimate_task_bytes(task)
        self._task_weight[id(task)] = weight
        blocked_at = perf_counter()
        try:
            # Byte-budget admission first (memory backpressure), then the count-bounded put.
            await self.budgets[node_id].acquire(weight)
        except BaseException:
            self._enqueued_at.pop(id(task), None)
            self._task_weight.pop(id(task), None)
            async with self._active_condition:
                self._active_tasks -= 1
                self._active_condition.notify_all()
            raise
        try:
            await self.queues[node_id].put(task)
        except BaseException:
            self._enqueued_at.pop(id(task), None)
            await self.budgets[node_id].release(self._task_weight.pop(id(task), 0))
            async with self._active_condition:
                self._active_tasks -= 1
                self._active_condition.notify_all()
            raise
        blocked_ms = (perf_counter() - blocked_at) * 1000
        await self.events.task_enqueued(node_id, task, self.queues[node_id].qsize())
        return blocked_ms

    async def _mark_task_done(self, task: Task) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._active_tasks -= 1
            if self._active_tasks <= 0:
                self._active_condition.notify_all()
        if self.lineages.tracks(task.lineage_id):
            await self.lineages.finish_task(task.lineage_id)

    async def _wait_until_idle(self) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            await self._active_condition.wait_for(lambda: self._active_tasks == 0 or bool(self._worker_errors))
        self.context.check_cancel()
        self._raise_worker_errors()

    async def _record_worker_error(self, error: BaseException) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._worker_errors.append(error)
            self._active_condition.notify_all()

    def _next_batch_index(self) -> int:
        self._batch_sequence += 1
        return self._batch_sequence

    def _total_memory_budget_bytes(self) -> int:
        budget_mb = self.context.config.memory_budget_mb
        if budget_mb is None or budget_mb <= 0:
            return DEFAULT_TOTAL_BUDGET_BYTES
        return int(budget_mb * 1024 * 1024)

    async def _worker(self, node: Node, worker_index: int, resource_priority: int) -> None:
        queue = self.queues[node.id]
        while True:
            first = await queue.get()
            batches, consumed_tasks = await self._collect_batches(node, first, queue)
            await self.events.queue_depth(node.id, queue.qsize())

            try:
                resource_policy = node.runtime.resource_policy.to_policy()
                while batches:
                    batch = batches.pop(0)
                    batch_index = self._next_batch_index()
                    batch_started_at = perf_counter()
                    queue_wait_ms = sum(batch_started_at - self._enqueued_at[id(task)] for task in batch) * 1000 / len(batch)
                    await self.events.batch_started(node, worker_index, batch_index, batch, queue_wait_ms)
                    resource_wait_started_at = perf_counter()
                    async with self.resource_pool.lease(resource_policy, resource_priority):
                        resource_wait_ms = (perf_counter() - resource_wait_started_at) * 1000
                        self.context.check_cancel()
                        load_started_at = perf_counter()
                        await self.load_node(node.id)
                        load_ms = (perf_counter() - load_started_at) * 1000
                        self.context.check_cancel()
                        node.logger.info("batch starting worker=%s batch=%s size=%s", worker_index, batch_index, len(batch))
                        execute_started_at = perf_counter()
                        outputs = await self._execute_node(node, [task.inputs for task in batch])
                        execute_ms = (perf_counter() - execute_started_at) * 1000
                        self.context.check_cancel()
                        if node.IS_INPUT:
                            await self.events.input_items_remaining(node, node.remaining_items(self.context))
                        unload_started_at = perf_counter()
                        if resource_policy.unload_after_stage and not resource_policy.keep_loaded:
                            await self.unload_node(node.id)
                        unload_ms = (perf_counter() - unload_started_at) * 1000
                    route_started_at = perf_counter()
                    await self.output_router.route(node, batch, outputs, batch_index)
                    route_ms = (perf_counter() - route_started_at) * 1000
                    timings = {
                        "queue_wait_ms": queue_wait_ms,
                        "resource_wait_ms": resource_wait_ms,
                        "load_ms": load_ms,
                        "execute_ms": execute_ms,
                        "unload_ms": unload_ms,
                        "route_ms": route_ms,
                        "total_ms": (perf_counter() - batch_started_at) * 1000,
                    }
                    await self.events.batch_completed(node, worker_index, batch_index, batch, outputs, timings)
                    node.logger.info("batch completed worker=%s batch=%s input_items=%s output_items=%s", worker_index, batch_index, len(batch), len(outputs))
                    self.context.check_cancel()
                    if node.IS_INPUT and node.remaining_items(self.context) > 0:
                        await self._enqueue_input(node)
            except Exception as error:
                node.logger.exception("node worker failed")
                await self.events.node_failed(node, worker_index, error)
                await self._record_worker_error(error)
                raise
            finally:
                for task in consumed_tasks:
                    self._enqueued_at.pop(id(task))
                    await self.budgets[node.id].release(self._task_weight.pop(id(task), 0))
                    queue.task_done()
                    await self._mark_task_done(task)

    async def _execute_node(self, node: Node, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def run_execute() -> list[dict[str, Any]]:
            with cancellation_scope(self.context.cancel_token), route_output_to_logger(node.logger):
                return asyncio.run(node.execute(inputs, self.context))

        return await asyncio.to_thread(run_execute)

    async def _collect_batches(self, node: Node, first: Task, queue: asyncio.Queue[Task]) -> tuple[list[list[Task]], list[Task]]:
        tasks = [first]
        batch_policy = node.runtime.batch_policy.to_policy()
        target_size = min(batch_policy.max_size, batch_policy.preferred_size)
        timeout = batch_policy.timeout_ms / 1000.0

        while len(tasks) < target_size:
            try:
                if timeout > 0:
                    tasks.append(await asyncio.wait_for(queue.get(), timeout=timeout))
                else:
                    tasks.append(queue.get_nowait())
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break
        while timeout > 0 and len(tasks) < node.runtime.queue_max_size:
            try:
                tasks.append(await asyncio.wait_for(queue.get(), timeout=timeout))
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break
        # Cap each batch to the node's memory slice so a batch of big payloads shrinks
        # below the count cap automatically (weights were already measured at enqueue).
        budget = self.budgets.get(node.id)
        byte_target = budget.budget_bytes if budget is not None else None
        batches = self.batch_planner.build_batches(
            tasks, batch_policy, weight_of=lambda task: self._task_weight.get(id(task), 0), byte_target=byte_target
        )
        return batches, tasks

    async def _enqueue_input(self, node: Node) -> None:
        await self._enqueue(
            node.id,
            Task(
                node_id=node.id,
                inputs={},
                input_packets={},
                lineage_id=f"input:{node.id}:{self.context.window_index}",
            ),
        )
