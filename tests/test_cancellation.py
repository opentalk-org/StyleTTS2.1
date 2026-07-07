from __future__ import annotations

import asyncio
import threading
import time
import unittest

from runflow.core.context import ExecutionContext
from runflow.core.graph import Graph
from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.types import DataType
from runflow.runtime.cancellation import check_cancel
from runflow.runtime.scheduler import WindowedScheduler


TEST_TYPE = DataType("test", object)


class BlockingInputNode(Node):
    NODE_TYPE = "BlockingInput"
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"value": Port("value", TEST_TYPE, mode=PortMode.STREAM)}

    def remaining_items(self, context):
        return 1

    async def execute(self, batch, context):
        while True:
            check_cancel()
            time.sleep(0.01)


class NonCooperativeBlockingInputNode(Node):
    NODE_TYPE = "NonCooperativeBlockingInput"
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"value": Port("value", TEST_TYPE, mode=PortMode.STREAM)}
    release = threading.Event()

    def remaining_items(self, context):
        return 1

    async def execute(self, batch, context):
        self.release.wait(10)
        return []


class CancellationTests(unittest.TestCase):
    def test_scheduler_cancel_reaches_context_local_check_in_blocking_node(self) -> None:
        async def scenario() -> None:
            graph = Graph()
            graph.add_node(BlockingInputNode("source"))
            scheduler = WindowedScheduler(graph, ExecutionContext(run_id="cancel-test"))
            task = asyncio.create_task(scheduler.arun())
            await asyncio.sleep(0.05)

            scheduler.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)

        asyncio.run(scenario())

    def test_scheduler_task_cancel_unblocks_non_cooperative_thread_node(self) -> None:
        async def scenario() -> None:
            NonCooperativeBlockingInputNode.release.clear()
            graph = Graph()
            graph.add_node(NonCooperativeBlockingInputNode("source"))
            scheduler = WindowedScheduler(graph, ExecutionContext(run_id="cancel-test"))
            task = asyncio.create_task(scheduler.arun())
            await asyncio.sleep(0.05)

            scheduler.cancel()
            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            NonCooperativeBlockingInputNode.release.set()

        try:
            asyncio.run(scenario())
        finally:
            NonCooperativeBlockingInputNode.release.set()


if __name__ == "__main__":
    unittest.main()
