from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioSegment


class MergeAlignmentSettings(StrictSettings):
    # Two words are treated as the same word (a duplicate to collapse) when they
    # share text and their spans are within this many seconds of each other.
    dedupe_window_sec: float = Field(default=0.2, ge=0.0, le=2.0, title="Dedupe window (s)")


class MergeAlignmentNode(Node):
    """Merge the per-word alignment of two audios of the same recording.

    Segments and text come from ``audio_a``; each segment's word alignment is the
    best combination of A's own words and B's words falling in that segment. Words
    are de-duplicated (same word at nearly the same time) keeping the higher-scored
    one, so overlapping aligners contribute their best timings without doubling up.
    """

    NODE_TYPE = "MergeAlignment"
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
        in_span = [word for word in other_words if seg.start <= _midpoint(word) <= seg.end]
        merged = _merge_words([*(seg.alignment or []), *in_span], self.settings.dedupe_window_sec)
        return replace(seg, alignment=merged)


def _merge_words(words: list[dict[str, Any]], window_sec: float) -> list[dict[str, Any]] | None:
    ordered = sorted(words, key=lambda word: (float(word["start"]), float(word["end"])))
    merged: list[dict[str, Any]] = []
    for word in ordered:
        duplicate = next((i for i, kept in enumerate(merged) if _same_word(kept, word, window_sec)), None)
        if duplicate is None:
            merged.append(dict(word))
        elif _score(word) > _score(merged[duplicate]):
            merged[duplicate] = dict(word)
    merged.sort(key=lambda word: float(word["start"]))
    return merged or None


def _same_word(a: dict[str, Any], b: dict[str, Any], window_sec: float) -> bool:
    if _normalized(a["word"]) != _normalized(b["word"]):
        return False
    overlaps = float(a["start"]) < float(b["end"]) and float(b["start"]) < float(a["end"])
    return overlaps or abs(_midpoint(a) - _midpoint(b)) <= window_sec


def _normalized(word: str) -> str:
    return "".join(char for char in str(word).lower() if char.isalnum())


def _midpoint(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2


def _score(word: dict[str, Any]) -> float:
    score = word.get("score")
    return float(score) if score is not None else -1.0
