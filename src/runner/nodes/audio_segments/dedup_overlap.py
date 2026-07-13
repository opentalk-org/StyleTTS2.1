from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio
from runner.nodes.statistics.segments import (
    DEFAULT_ALIGNMENT_MATCH_WINDOW_SEC,
    DEFAULT_MIN_OVERLAP_RATIO,
    deduplicate_overlapping_segments,
)


class DeduplicateOverlappingSegmentsSettings(StrictSettings):
    min_overlap_ratio: float = Field(default=DEFAULT_MIN_OVERLAP_RATIO, ge=0.0, le=1.0)
    alignment_match_window_sec: float = Field(
        default=DEFAULT_ALIGNMENT_MATCH_WINDOW_SEC,
        ge=0.0,
        le=2.0,
        title="Preferred alignment match window (s)",
    )


class DeduplicateOverlappingSegmentsNode(Node):
    NODE_TYPE = "DeduplicateOverlappingSegments"
    DESCRIPTION = "Collapse duplicate transcript segments that cover almost the same span of audio, keeping the consensus segment and merging word alignments from every member. Matching words are paired one-to-one across aligners even when timings are misaligned, and the higher-scored timing wins without multiplying repeated words."
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
                alignment_match_window_sec=self.settings.alignment_match_window_sec,
            )
            metadata = {**audio.metadata, "overlap_segments_collapsed": collapsed}
            outputs.append({"audio": replace(audio, segments=kept, metadata=metadata)})
        return outputs
