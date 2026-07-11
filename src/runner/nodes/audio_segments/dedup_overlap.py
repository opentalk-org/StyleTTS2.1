from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio
from runner.nodes.statistics.segments import DEFAULT_MIN_OVERLAP_RATIO, deduplicate_overlapping_segments


class DeduplicateOverlappingSegmentsSettings(StrictSettings):
    min_overlap_ratio: float = Field(default=DEFAULT_MIN_OVERLAP_RATIO, ge=0.0, le=1.0)


class DeduplicateOverlappingSegmentsNode(Node):
    NODE_TYPE = "DeduplicateOverlappingSegments"
    DESCRIPTION = "Collapse duplicate transcript segments that cover almost the same span of audio, keeping a single segment where several overlap. Takes audio with segments and outputs the same audio with the redundant overlapping segments removed (the count collapsed is recorded in metadata). Use it to clean up after multiple transcribers or aligners have annotated the same recording. Tune the minimum overlap ratio to control how much two segments must overlap before they are treated as duplicates."
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
            outputs.append({"audio": replace(audio, segments=kept, metadata=metadata)})
        return outputs
