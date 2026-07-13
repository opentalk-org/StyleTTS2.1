from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.audio_segments.alignment_merge import alignment_midpoint, merge_alignment_tracks
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioSegment


class MergeAlignmentSettings(StrictSettings):
    dedupe_window_sec: float = Field(default=0.2, ge=0.0, le=2.0, title="Preferred match window (s)")


class MergeAlignmentNode(Node):
    NODE_TYPE = "MergeAlignment"
    DESCRIPTION = "Merge the per-word alignment of two versions of the same recording. Segments and text come from the first audio input; each segment's word timings are the best combination of its own words and the second input's words that fall within it. Matching words are paired one-to-one across aligners even when their timings are misaligned, while genuine repetitions within one alignment are preserved. The higher-scored timing wins each pair."
    CATEGORY = "Audio"
    SETTINGS = MergeAlignmentSettings
    INPUTS = {"audio_a": AudioPort(), "audio_b": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            context.check_cancel()
            audio_a: Audio = inputs["audio_a"]
            audio_b: Audio = inputs["audio_b"]
            outputs.append({"audio": self._merge(audio_a, audio_b)})
        return outputs

    def _merge(self, audio_a: Audio, audio_b: Audio) -> Audio:
        other_words = [word for seg in audio_b.segments for word in (seg.alignment or [])]
        segments = [self._merge_segment(seg, other_words) for seg in audio_a.segments]
        return replace(audio_a, segments=segments)

    def _merge_segment(self, seg: AudioSegment, other_words: list[dict[str, Any]]) -> AudioSegment:
        in_span = [word for word in other_words if seg.start <= alignment_midpoint(word) <= seg.end]
        merged = merge_alignment_tracks(
            [seg.alignment or [], in_span],
            self.settings.dedupe_window_sec,
        )
        return replace(seg, alignment=merged)
