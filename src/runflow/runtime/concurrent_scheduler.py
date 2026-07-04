from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.task import Packet, Task, lineage_from_value, metadata_from_value
from runflow.planning.batch_planner import BatchPlanner
from runflow.planning.graph_validator import GraphValidator
from runflow.runtime.artifact_store import ArtifactStore
from runflow.runtime.concurrent_node_manager import ConcurrentNodeManager
from runflow.runtime.join_builder import pop_ready_join_tasks
from runflow.runtime.resource_pool import ResourcePool
from runflow.runtime.window_manager import WindowManager


def _is_stream_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict, Path)):
        return False
    return isinstance(value, Iterable)


class ConcurrentWindowedScheduler:
    """Concurrent, bounded-queue, windowed scheduler.

    Nodes declare generic ResourcePolicy requirements, and ResourcePool decides
    which workers can run at the same time.
    """

    def __init__(self, graph: Graph, context: ExecutionContext):
        self.graph = graph
        self.context = context
        self.validator = GraphValidator()
        self.batch_planner = BatchPlanner()
        self.node_manager = ConcurrentNodeManager(context)
        self.artifact_store = ArtifactStore(context.work_dir / context.run_id)

        limits = dict(context.config.get("resources", {}))
        if not limits:
            limits = {
                "io": 2,
                "cpu_workers": 4,
                "accelerator": 1,
                "vram_gb": 12,
            }
        self.resource_pool = ResourcePool(limits=limits)

        self.queues: dict[str, asyncio.Queue[Task]] = {}
        self.join_buffers: dict[tuple[str, str], dict[str, list[Packet]]] = defaultdict(lambda: defaultdict(list))
        self.queue_max_size = max(1, int(context.config.get("queue_max_size", 128)))
        self._active_tasks = 0
        self._active_condition: asyncio.Condition | None = None
        self._workers: list[asyncio.Task] = []

    def run(self) -> None:
        asyncio.run(self.arun())

    async def arun(self) -> None:
        self.validator.validate(self.graph)
        input_items = self.context.input_items or self._discover_source_items()
        windows = WindowManager.from_config(input_items, self.context.config.get("window", {}))

        try:
            for window_index, items in enumerate(windows.iter_windows()):
                self.context.window_index = window_index
                self.context.current_window_items = items
                print(f"\n=== Concurrent window {window_index}: {len(items)} item(s) ===")
                await self._run_window()

            self.artifact_store.write_index()
        finally:
            await self._stop_workers()
            await self.node_manager.unload_all()

    def _discover_source_items(self) -> list[Any]:
        items: list[Any] = []
        for node in self.graph.source_nodes():
            list_items = getattr(node, "list_items", None)
            if callable(list_items):
                items.extend(list_items())
                continue

            # Convenience fallback for path-listing source nodes. Runtime remains
            # generic; this avoids forcing existing path-based examples to change.
            list_paths = getattr(node, "list_paths", None)
            if callable(list_paths):
                items.extend(list_paths())
        return items

    async def _run_window(self) -> None:
        self.queues = {node_id: asyncio.Queue(maxsize=self.queue_max_size) for node_id in self.graph.nodes}
        self.join_buffers.clear()
        self._active_tasks = 0
        self._active_condition = asyncio.Condition()
        self._workers = []

        for node in self.graph.nodes.values():
            concurrency = int(
                node.params.get(
                    "concurrency",
                    self.context.config.get("node_concurrency", {}).get(node.id, 1),
                )
            )
            concurrency = max(1, concurrency)
            for worker_index in range(concurrency):
                self._workers.append(
                    asyncio.create_task(self._worker(node, worker_index), name=f"{node.id}:{worker_index}")
                )

        for source in self.graph.source_nodes():
            if source.INPUTS:
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

    async def _enqueue(self, node_id: str, task: Task) -> None:
        if self._active_condition is None:
            raise RuntimeError("Scheduler is not running")
        async with self._active_condition:
            self._active_tasks += 1
        await self.queues[node_id].put(task)

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
            await self._active_condition.wait_for(lambda: self._active_tasks == 0)

    async def _worker(self, node: Node, worker_index: int) -> None:
        queue = self.queues[node.id]
        while True:
            first = await queue.get()
            batches, consumed_tasks = await self._collect_batches(node, first, queue)

            try:
                async with self.resource_pool.lease(node.RESOURCE_POLICY):
                    await self.node_manager.ensure_loaded(node)
                    for batch in batches:
                        print(
                            f"[{node.id}#{worker_index}] {node.NODE_TYPE}: "
                            f"batch={len(batch)} resources={node.RESOURCE_POLICY.requirements()}"
                        )
                        try:
                            outputs = await asyncio.to_thread(
                                node.execute,
                                [task.inputs for task in batch],
                                self.context,
                            )
                            await self._route_outputs(node, batch, outputs)
                        finally:
                            if node.RESOURCE_POLICY.unload_after_stage and not node.RESOURCE_POLICY.keep_loaded:
                                await self.node_manager.unload(node)
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
        max_size = max(1, node.BATCH_POLICY.max_size)
        preferred_size = max(1, node.BATCH_POLICY.preferred_size)
        target_size = min(max_size, preferred_size)
        timeout = max(0, node.BATCH_POLICY.timeout_ms) / 1000.0

        while len(tasks) < target_size:
            try:
                if timeout > 0:
                    tasks.append(await asyncio.wait_for(queue.get(), timeout=timeout))
                else:
                    tasks.append(queue.get_nowait())
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        return self.batch_planner.build_batches(tasks, node.BATCH_POLICY), tasks

    async def _route_outputs(self, node: Node, batch: list[Task], outputs: list[dict[str, Any]]) -> None:
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
                if port_name not in node.OUTPUTS:
                    raise KeyError(f"{node.id} returned undeclared output port: {port_name}")

                port = node.OUTPUTS[port_name]
                if port.mode == PortMode.STREAM and _is_stream_iterable(value):
                    values = list(value)
                else:
                    values = [value]

                for item in values:
                    packet = Packet(
                        node_id=node.id,
                        port=port_name,
                        dtype=port.dtype.name,
                        value=item,
                        lineage_id=lineage_from_value(item, inherited=task.lineage_id),
                        metadata={**task.metadata, **metadata_from_value(item)},
                    )
                    self.artifact_store.register_packet(packet)
                    await self._deliver_packet(packet)

    async def _deliver_packet(self, packet: Packet) -> None:
        for edge in self.graph.outgoing_edges(packet.node_id):
            if edge.source.port != packet.port:
                continue

            target_node = self.graph.nodes[edge.target.node_id]
            target_port = edge.target.port

            if self._can_create_single_input_task(target_node):
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

            tasks = self._add_to_join_buffer(target_node, target_port, packet)
            for task in tasks:
                await self._enqueue(target_node.id, task)

    def _can_create_single_input_task(self, node: Node) -> bool:
        required = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
        return len(node.INPUTS) == 1 and len(required) == 1

    def _add_to_join_buffer(self, node: Node, input_name: str, packet: Packet) -> list[Task]:
        key = (node.id, packet.lineage_id)
        grouped = self.join_buffers[key]
        grouped[input_name].append(packet)

        required = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
        if not all(name in grouped or name in node.params for name in required):
            return []

        tasks = pop_ready_join_tasks(node, packet.lineage_id, grouped)
        if not any(grouped.values()):
            del self.join_buffers[key]
        return tasks
