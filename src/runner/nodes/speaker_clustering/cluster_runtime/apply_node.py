from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import SaveResultPort, SpeakerAuditRefPort
from runner.nodes.models import SpeakerAuditRef
from runner.nodes.speaker_clustering.cluster_runtime.apply import apply_speaker_audit


class ApplySpeakerClustersSettings(StrictSettings):
    audio_page_size: int = Field(default=500, gt=0, le=10_000)
    assignment_batch_rows: int = Field(default=100_000, gt=0)


@dataclass(frozen=True)
class ApplyProgressReporter:
    loop: asyncio.AbstractEventLoop
    context: Any
    node_id: str

    def report(self, current: int, total: int, message: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.context.report_progress(self.node_id, current, total, message),
            self.loop,
        )
        future.result()


class ApplySpeakerClustersNode(Node):
    NODE_TYPE = "ApplySpeakerClusters"
    DESCRIPTION = "Create voices and apply only accepted assignments from a completed audit."
    CATEGORY = "Speaker Clustering"
    SETTINGS = ApplySpeakerClustersSettings
    INPUTS = {"audit": SpeakerAuditRefPort()}
    OUTPUTS = {"save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=False)
    QUEUE_MAX_SIZE = 1

    async def execute(
        self, batch: list[dict[str, Any]], context: Any
    ) -> list[dict[str, Any]]:
        if len(batch) != 1:
            raise ValueError(f"{self.id} requires exactly one completed speaker audit")
        audit = batch[0]["audit"]
        assert isinstance(audit, SpeakerAuditRef)
        reporter = ApplyProgressReporter(asyncio.get_running_loop(), context, self.id)
        result = await asyncio.to_thread(
            apply_speaker_audit,
            audit,
            context.node_dir(self.id) / f"assignments-{audit.audit_id}.sqlite3",
            self.settings.audio_page_size,
            self.settings.assignment_batch_rows,
            context.check_cancel,
            reporter.report,
        )
        return [{"save_result": result, INPUT_INDEX_OUTPUT: 0}]
