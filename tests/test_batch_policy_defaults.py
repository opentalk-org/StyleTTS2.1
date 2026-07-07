from __future__ import annotations

import unittest

from runflow.core.node import Node
from runflow.core.task import Task
from runflow.planning.batch_planner import BatchPlanner
from runflow.policies import BatchMode


class PlainBatchNode(Node):
    NODE_TYPE = "PlainBatch"

    async def execute(self, batch, context):
        return []


class BatchPolicyDefaultTests(unittest.TestCase):
    def test_plain_node_defaults_to_sixty_four_item_micro_batches(self) -> None:
        node = PlainBatchNode("plain")

        policy = node.runtime.batch_policy

        self.assertEqual(policy.mode, BatchMode.MICRO_BATCH)
        self.assertEqual(policy.preferred_size, 64)
        self.assertEqual(policy.max_size, 64)

    def test_batch_planner_uses_sixty_four_item_default_policy(self) -> None:
        node = PlainBatchNode("plain")
        tasks = [
            Task(node_id="plain", inputs={"value": index}, input_packets={}, lineage_id=str(index))
            for index in range(130)
        ]

        batches = BatchPlanner().build_batches(tasks, node.runtime.batch_policy.to_policy())

        self.assertEqual([len(batch) for batch in batches], [64, 64, 2])


if __name__ == "__main__":
    unittest.main()
