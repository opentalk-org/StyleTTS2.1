from __future__ import annotations

from runflow.core.graph import Graph


class BranchPlanner:
    """Placeholder for branch-order planning.

    Later you can choose whether Whisper, Parakeet, and Canary run serially,
    concurrently, or on different devices.
    """

    def annotate(self, graph: Graph) -> Graph:
        return graph
