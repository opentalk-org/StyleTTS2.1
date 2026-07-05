from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.task import Packet, Task, lineage_from_value, metadata_from_value
from runflow.planning.batch_planner import BatchPlanner
from runflow.planning.graph_validator import GraphValidator
from runflow.runtime.artifact_store import ArtifactStore
from runflow.runtime.input_progress import ensure_input_progress, has_remaining_items, processed_counts, remaining_counts
from runflow.runtime.node_manager import NodeManager
from runflow.runtime.output_values import output_values
from runflow.runtime.resource_pool import ResourcePool
from runflow.runtime.routing import add_to_join_buffer, can_create_single_input_task
from runflow.runtime.scheduler_events import SchedulerEventEmitter


class WindowedScheduler:
    def __init__(self, graph: Graph, context: ExecutionContext):
        self.graph = graph
        self.context = context
        self.validator = GraphValidator()
        self.events = SchedulerEventEmitter(context)
        self.batch_planner = BatchPlanner()
        self.node_manager = NodeManager(context)
        self.artifact_store = ArtifactStore(context.work_dir / context.run_id)

        self.resource_pool = ResourcePool(limits=dict(context.config.resources))

        self.queues: dict[str, asyncio.Queue[Task]] = {}
        self.join_buffers: dict[tuple[str, str], dict[str, list[Packet]]] = defaultdict(lambda: defaultdict(list))
        self._active_tasks = 0
        self._active_condition: asyncio.Condition | None = None
        self._workers: list[asyncio.Task] = []
        self._worker_errors: list[BaseException] = []
        self._batch_sequence = 0

    def run(self) -> None:
        asyncio.run(self.arun())

    async def arun(self) -> None:
        self.validator.validate(self.graph)
        input_nodes = self.graph.input_nodes()
        counts = remaining_counts(input_nodes, self.context)
        total_items = sum(counts.values())
        await self.events.run_started(total_items, self.graph.nodes.keys())
        for node in input_nodes:
            await self.events.input_items_discovered(node, counts[node.id])
            await self.events.input_items_remaining(node, counts[node.id])

        try:
            window_index = 0
            while has_remaining_items(counts):
                self.context.window_index = window_index
                await self.events.window_started(window_index, sum(counts.values()), counts)
                await self._run_window(input_nodes, counts)
                next_counts = remaining_counts(input_nodes, self.context)
                ensure_input_progress(counts, next_counts)
                processed = processed_counts(counts, next_counts)
                await self.events.window_completed(window_index, sum(processed.values()), processed)
                counts = next_counts
                window_index += 1

            self.artifact_store.write_index()
            await self.context.emit_event("run_completed", message="run completed")
        finally:
            await self._stop_workers()
            await self.node_manager.unload_all()

    async def _run_window(self, input_nodes: list[Node], remaining_counts: dict[str, int]) -> None:
        self.queues = {
            node_id: asyncio.Queue(maxsize=node.runtime.queue_max_size)
            for node_id, node in self.graph.nodes.items()
        }
        self.join_buffers.clear()
        self._active_tasks = 0
        self._active_condition = asyncio.Condition()
        self._workers = []
        self._worker_errors = []
        self._batch_sequence = 0

        for node in self.graph.nodes.values():
            self._workers.append(asyncio.create_task(self._worker(node, 0), name=f"{node.id}:0"))

        for source in input_nodes:
            if remaining_counts[source.id] <= 0:
                continue
            await self._enqueue(
                source.id,
                Task(
                    node_id=source.id,
                    inputs={},
                    input_packets={},
                    lineage_id=f"window:{self.context.window_index}",
                ),
            )

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

    async def _enqueue(self, node_id: str, task: Task) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._active_tasks += 1
        await self.queues[node_id].put(task)
        await self.events.task_enqueued(node_id, task, self.queues[node_id].qsize())

    async def _mark_task_done(self) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._active_tasks -= 1
            if self._active_tasks <= 0:
                self._active_condition.notify_all()

    async def _wait_until_idle(self) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            await self._active_condition.wait_for(lambda: self._active_tasks == 0 or bool(self._worker_errors))
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

    async def _worker(self, node: Node, worker_index: int) -> None:
        queue = self.queues[node.id]
        while True:
            first = await queue.get()
            batches, consumed_tasks = await self._collect_batches(node, first, queue)
            await self.events.queue_depth(node.id, queue.qsize())

            try:
                resource_policy = node.runtime.resource_policy.to_policy()
                async with self.resource_pool.lease(resource_policy):
                    was_loaded = node.id in self.node_manager.loaded
                    await self.node_manager.ensure_loaded(node)
                    if not was_loaded and node.id in self.node_manager.loaded:
                        await self.events.node_loaded(node)
                    for batch in batches:
                        batch_index = self._next_batch_index()
                        await self.events.batch_started(node, worker_index, batch_index, batch)
                        try:
                            outputs = await node.execute([task.inputs for task in batch], self.context)
                            if node.IS_INPUT:
                                await self.events.input_items_remaining(node, node.remaining_items(self.context))
                            await self._route_outputs(node, batch, outputs, batch_index)
                            await self.events.batch_completed(node, worker_index, batch_index, batch, outputs)
                        finally:
                            if resource_policy.unload_after_stage and not resource_policy.keep_loaded:
                                await self.node_manager.unload(node)
                                await self.events.node_unloaded(node)
            except Exception as error:
                await self.events.node_failed(node, worker_index, error)
                await self._record_worker_error(error)
                raise
            finally:
                for _ in consumed_tasks:
                    queue.task_done()
                    await self._mark_task_done()

    async def _collect_batches(
        self,
        node: Node,
        first: Task,
        queue: asyncio.Queue[Task],
    ) -> tuple[list[list[Task]], list[Task]]:
        tasks = [first]
        batch_policy = node.runtime.batch_policy.to_policy()
        max_size = batch_policy.max_size
        preferred_size = batch_policy.preferred_size
        target_size = min(max_size, preferred_size)
        timeout = batch_policy.timeout_ms / 1000.0

        while len(tasks) < target_size:
            try:
                if timeout > 0:
                    tasks.append(await asyncio.wait_for(queue.get(), timeout=timeout))
                else:
                    tasks.append(queue.get_nowait())
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        return self.batch_planner.build_batches(tasks, batch_policy), tasks

    async def _route_outputs(
        self,
        node: Node,
        batch: list[Task],
        outputs: list[dict[str, Any]],
        batch_index: int,
    ) -> None:
        if len(outputs) == len(batch):
            task_for_output = batch
        elif len(batch) == 1:
            task_for_output = [batch[0] for _ in outputs]
        else:
            raise ValueError(
                f"{node.id} returned {len(outputs)} output item(s) for batch size {len(batch)}"
            )

        for task, output_dict in zip(task_for_output, outputs):
            for port_name, value in output_dict.items():
                if port_name == "__progress__":
                    await self.events.node_progress(node, value, batch_index)
                    continue
                if port_name not in node.OUTPUTS:
                    raise KeyError(f"{node.id} returned undeclared output port: {port_name}")

                port = node.OUTPUTS[port_name]
                for item in output_values(node, port_name, port, value):
                    packet = Packet(
                        node_id=node.id,
                        port=port_name,
                        dtype=port.dtype.name,
                        value=item,
                        lineage_id=lineage_from_value(item, inherited=task.lineage_id),
                        metadata={**task.metadata, **metadata_from_value(item)},
                    )
                    self.artifact_store.register_packet(packet)
                    await self.events.packet_created(packet, batch_index)
                    await self._deliver_packet(packet)

    async def _deliver_packet(self, packet: Packet) -> None:
        for edge in self.graph.outgoing_edges(packet.node_id):
            if edge.source.port != packet.port:
                continue

            target_node = self.graph.nodes[edge.target.node_id]
            target_port = edge.target.port
            await self.events.packet_delivered(packet, target_node, target_port)

            if can_create_single_input_task(target_node):
                await self._enqueue(
                    target_node.id,
                    Task(
                        node_id=target_node.id,
                        inputs={target_port: packet.value},
                        input_packets={target_port: packet},
                        lineage_id=packet.lineage_id,
                        metadata=packet.metadata,
                    ),
                )
                continue

            tasks = add_to_join_buffer(target_node, target_port, packet, self.join_buffers)
            if not tasks:
                await self.events.join_waiting(target_node, target_port, packet)
            for task in tasks:
                await self.events.join_ready(task)
                await self._enqueue(target_node.id, task)
