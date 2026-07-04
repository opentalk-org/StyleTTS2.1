from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from runflow.core.context import ExecutionContext
from runflow.core.graph import Edge, Graph
from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.task import Packet, Task, lineage_from_value, metadata_from_value
from runflow.planning.batch_planner import BatchPlanner
from runflow.planning.graph_validator import GraphValidator
from runflow.planning.stage_builder import StageBuilder
from runflow.runtime.artifact_store import ArtifactStore
from runflow.runtime.join_builder import build_join_tasks
from runflow.runtime.node_manager import NodeManager
from runflow.runtime.window_manager import WindowManager


def _is_stream_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict, Path)):
        return False
    return isinstance(value, Iterable)


class WindowedBatchScheduler:
    """Windowed, stage-batched executor.

    It processes a bounded input window, runs each node in topological order,
    batches per node using that node's BatchPolicy, and uses NodeManager to
    avoid node setup switching per item.
    """

    def __init__(self, graph: Graph, context: ExecutionContext):
        self.graph = graph
        self.context = context
        self.validator = GraphValidator()
        self.stage_builder = StageBuilder()
        self.batch_planner = BatchPlanner()
        self.node_manager = NodeManager(context)
        self.artifact_store = ArtifactStore(context.work_dir / context.run_id)

    def run(self) -> None:
        self.validator.validate(self.graph)
        input_items = self.context.input_items or self._discover_source_items()
        windows = WindowManager.from_config(input_items, self.context.config.get("window", {}))
        stages = self.stage_builder.build(self.graph)

        try:
            for window_index, items in enumerate(windows.iter_windows()):
                self.context.window_index = window_index
                self.context.current_window_items = items
                state: dict[tuple[str, str], list[Packet]] = defaultdict(list)

                print(f"\n=== Window {window_index}: {len(items)} item(s) ===")
                for stage in stages:
                    for node in stage.nodes:
                        tasks = self._build_tasks_for_node(node, state)
                        if not tasks:
                            continue

                        self.node_manager.ensure_loaded(node)
                        batches = self.batch_planner.build_batches(tasks, node.BATCH_POLICY)

                        print(
                            f"[{node.id}] {node.NODE_TYPE}: "
                            f"{len(tasks)} task(s), {len(batches)} batch(es), "
                            f"batch_policy={node.BATCH_POLICY.preferred_size}"
                        )

                        for batch in batches:
                            self._execute_node_batch(node, batch, state)

                        if node.RESOURCE_POLICY.unload_after_stage:
                            self.node_manager.unload(node)

            self.artifact_store.write_index()
        finally:
            self.node_manager.unload_all()

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

    def _build_tasks_for_node(self, node: Node, state: dict[tuple[str, str], list[Packet]]) -> list[Task]:
        incoming = self.graph.incoming_edges(node.id)

        # Source node.
        if not incoming and not node.INPUTS:
            return [
                Task(
                    node_id=node.id,
                    inputs={},
                    input_packets={},
                    lineage_id=f"window:{self.context.window_index}",
                )
            ]

        by_target_port: dict[str, list[Packet]] = defaultdict(list)
        for edge in incoming:
            by_target_port[edge.target.port].extend(state.get((edge.source.node_id, edge.source.port), []))

        # Single-input nodes should run once per packet. This also handles many
        # upstream branches feeding the same input, for example three ASR nodes
        # feeding SaveTranscript.transcript.
        required_input_names = [name for name, port in node.INPUTS.items() if not port.optional and name not in node.params]
        if len(required_input_names) == 1 and len(node.INPUTS) == 1:
            port_name = required_input_names[0]
            tasks = []
            for packet in by_target_port.get(port_name, []):
                tasks.append(
                    Task(
                        node_id=node.id,
                        inputs={port_name: packet.value},
                        input_packets={port_name: packet},
                        lineage_id=packet.lineage_id,
                        metadata=packet.metadata,
                    )
                )
            return tasks

        # Multi-input join by lineage id.
        lineages: dict[str, dict[str, list[Packet]]] = defaultdict(lambda: defaultdict(list))
        for input_name, packets in by_target_port.items():
            for packet in packets:
                lineages[packet.lineage_id][input_name].append(packet)

        tasks: list[Task] = []
        for lineage_id, grouped in lineages.items():
            if not all(name in grouped or name in node.params for name in required_input_names):
                continue

            tasks.extend(build_join_tasks(node, lineage_id, grouped))

        return tasks

    def _execute_node_batch(self, node: Node, batch: list[Task], state: dict[tuple[str, str], list[Packet]]) -> None:
        outputs = node.execute([task.inputs for task in batch], self.context)

        task_for_output: list[Task]
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
                values: list[Any]
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
                    state[(node.id, port_name)].append(packet)
                    self.artifact_store.register_packet(packet)
