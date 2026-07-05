from __future__ import annotations

from runflow.tmp_nodes.asr.base import ASRNode
from runflow.policies import BatchMode, BatchPolicy


class CanaryNode(ASRNode):
    NODE_TYPE = "Canary"
    MODEL_NAME = "canary"
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=64,
        max_size=64,
        timeout_ms=1000,
        sort_by="duration",
    )
