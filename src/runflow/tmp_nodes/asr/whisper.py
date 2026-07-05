from __future__ import annotations

from runflow.tmp_nodes.asr.base import ASRNode
from runflow.policies import BatchMode, BatchPolicy


class WhisperNode(ASRNode):
    NODE_TYPE = "Whisper"
    MODEL_NAME = "whisper"
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=64,
        max_size=64,
        timeout_ms=1000,
        sort_by="duration",
    )
