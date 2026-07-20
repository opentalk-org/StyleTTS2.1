from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio
from runner.nodes.statistics.segments import (
    DEFAULT_MIN_OVERLAP_RATIO,
    deduplicate_overlapping_segments,
)


class DeduplicateOverlappingSegmentsSettings(StrictSettings):
    min_overlap_ratio: float = Field(default=DEFAULT_MIN_OVERLAP_RATIO, ge=0.0, le=1.0)


class DeduplicateOverlappingSegmentsNode(Node):
    NODE_TYPE = "DeduplicateOverlappingSegments"
    DESCRIPTION = (
        "Collapse overlapping duplicate transcripts into the consensus segment. Its alignment exactly follows "
        "the selected transcript: member timings are matched in word order, higher-scored timings win, extra "
        "words are discarded, and missing words receive bounded interpolated timings."
    )
    CATEGORY = "Audio"
    SETTINGS = DeduplicateOverlappingSegmentsSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            kept, collapsed = deduplicate_overlapping_segments(
                audio.segments,
                min_overlap_ratio=self.settings.min_overlap_ratio,
            )
            metadata = {**audio.metadata, "overlap_segments_collapsed": collapsed}
            outputs.append({"audio": replace(audio, segments=kept, annotations=audio.annotations.model_copy(update={"metadata": metadata}))})
        return outputs
