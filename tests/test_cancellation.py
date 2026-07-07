from __future__ import annotations

import asyncio
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


if __name__ == "__main__":
    unittest.main()
