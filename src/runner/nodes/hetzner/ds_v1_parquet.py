from __future__ import annotations

from typing import Any

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.hetzner.ds_v1_pipeline import (
    DsV1AudioPipeline,
    HetznerDsV1ParquetAudioSourceSettings,
)
from runner.nodes.models import Audio


class HetznerDsV1ParquetAudioSourceNode(Node):
    NODE_TYPE = "HetznerDsV1ParquetAudioSource"
    DESCRIPTION = "Discover and prefetch ds_v1 Parquet files from Hetzner, convert Opus recordings to WAV concurrently, merge matching ds_v2 metadata, and stream absolute-time transcript segments."
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV1ParquetAudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 1

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._pipeline: DsV1AudioPipeline | None = None
        self._emitted = 0

    def remaining_items(self, context: Any) -> int:
        if self._pipeline is None:
            return self.settings.row_limit
        return self._pipeline.remaining

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        if self._pipeline is None:
            self._pipeline = DsV1AudioPipeline(self.settings, context, self.id)
        audio = self._pipeline.next_audio()
        if audio is None:
            return []
        self._emitted += 1
        await context.report_progress(
            self.id,
            self._emitted,
            self.settings.row_limit,
            f"converted {self._emitted}/{self.settings.row_limit} at {self._pipeline.realtime_factor:.1f}x realtime",
            {"realtime_factor": self._pipeline.realtime_factor},
        )
        return [{"audio": audio}]

    async def teardown(self, context: Any) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None


__all__ = ["HetznerDsV1ParquetAudioSourceNode", "HetznerDsV1ParquetAudioSourceSettings"]
