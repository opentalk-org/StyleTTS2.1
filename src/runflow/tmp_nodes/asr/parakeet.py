from __future__ import annotations

from runflow.tmp_nodes.asr.base import ASRNode
from runflow.policies import BatchMode, BatchPolicy


class ParakeetNode(ASRNode):
    NODE_TYPE = "Parakeet"
    MODEL_NAME = "parakeet"
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=16,
        max_size=24,
        sort_by="duration",
    )
