from __future__ import annotations

from runflow.core.task import Task
from runflow.planning.batch_planner import BatchPlanner
from runflow.policies import BatchPolicy


class BatchBuilder:
    def __init__(self):
        self.planner = BatchPlanner()

    def build(self, tasks: list[Task], policy: BatchPolicy) -> list[list[Task]]:
        return self.planner.build_batches(tasks, policy)
