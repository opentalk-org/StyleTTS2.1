from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AUDIO
from runner.nodes.models import Audio, AudioSegment, stable_id

Mode = Literal["create_new", "replace_all"]

_LENGTH_REF_SECONDS = 60.0


class PlanSegmentGroupsSettings(StrictSettings):
    mode: Mode = "create_new"
    min_total_seconds: float = Field(default=1.0, ge=0.0)
    min_total_ipa_chars: int = Field(default=1, ge=0)
    max_gap_seconds: float | None = Field(default=0.5, ge=0.0)
    max_ipa_chars: int = Field(default=512, ge=1)
    max_merged_duration_seconds: float | None = Field(default=None, ge=0.0)


class PlanSegmentGroupsNode(Node):
    NODE_TYPE = "PlanSegmentGroups"
    CATEGORY = "Audio"
    SETTINGS = PlanSegmentGroupsSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            source: Audio = inputs["audio"]
            ordered = sorted(source.segments, key=lambda segment: segment.start)
            groups = iter_split_groups(
                ordered,
                min_total_seconds=self.settings.min_total_seconds,
                min_ipa_at_index=lambda index: _sample_min_ipa_for_remaining(ordered, index, self.settings),
                max_gap_seconds=self.settings.max_gap_seconds,
                max_ipa_chars=self.settings.max_ipa_chars,
                max_merged_duration_seconds=self.settings.max_merged_duration_seconds,
            )
            group_count = len(groups)
            for index, segments in enumerate(groups):
                outputs.append({"audio": _planned_audio(source, segments, index, group_count, self.settings.mode)})
        return outputs


def segment_ipa_char_count(segment: AudioSegment) -> int:
    return len(segment.phon or "")


def sample_min_ipa_chars_from_audio_length(
    *,
    audio_length_seconds: float,
    min_total_ipa_chars: int,
    max_ipa_chars: int,
) -> int:
    length = max(0.0, float(audio_length_seconds))
    scale = min(1.0, length / _LENGTH_REF_SECONDS)
    floor = int(min_total_ipa_chars)
    ceiling = int(max_ipa_chars)
    span = max(0, ceiling - floor)
    dynamic_high = floor + int(round(span * scale))
    dynamic_high = max(floor, min(ceiling, dynamic_high))
    inner = float(dynamic_high - floor)
    mean = floor + inner / 2.0
    std = max(inner / 2.0, 1.0) if inner > 0 else 1.0
    sampled = int(round(random.gauss(mean, std)))
    return max(floor, min(ceiling, sampled))


def iter_split_groups(
    segments: list[AudioSegment],
    *,
    min_total_seconds: float,
    min_ipa_at_index: Callable[[int], int],
    max_gap_seconds: float | None,
    max_ipa_chars: int = 512,
    max_merged_duration_seconds: float | None = None,
) -> list[list[AudioSegment]]:
    ordered = sorted(segments, key=lambda segment: segment.start)
    if max_merged_duration_seconds is not None:
        limit = float(max_merged_duration_seconds)
        ordered = [segment for segment in ordered if segment.duration <= limit + 1e-12]

    groups: list[list[AudioSegment]] = []
    index = 0
    while index < len(ordered):
        reachable_end = _reachable_end(ordered, index, max_gap_seconds)
        best_end = _best_group_end(
            ordered,
            index,
            reachable_end,
            min_total_seconds,
            min_ipa_at_index(index),
            max_ipa_chars,
            max_merged_duration_seconds,
        )
        if best_end is None:
            best_end = _fallback_group_end(ordered, index, reachable_end, max_ipa_chars, max_merged_duration_seconds)
        if best_end is None:
            index += 1
            continue
        groups.append(ordered[index:best_end])
        index = best_end
    return groups


def group_total_duration(segments: list[AudioSegment]) -> float:
    return sum(segment.duration for segment in segments)


def group_span_seconds(segments: list[AudioSegment]) -> float:
    if not segments:
        return 0.0
    return float(max(segment.end for segment in segments) - min(segment.start for segment in segments))


def merge_segments_text_phon(segments: list[AudioSegment]) -> tuple[str, str]:
    texts = [segment.text or "" for segment in segments]
    phons = [segment.phon or "" for segment in segments]
    return "".join(texts), "".join(phons)


def _planned_audio(
    source: Audio,
    segments: list[AudioSegment],
    index: int,
    group_count: int,
    mode: Mode,
) -> Audio:
    merged_text, merged_phon = merge_segments_text_phon(segments)
    span_start = min(segment.start for segment in segments)
    span_end = max(segment.end for segment in segments)
    group_id = stable_id("audio_group", source.id, index, *(segment.id for segment in segments))
    metadata = {
        **source.metadata,
        "source_group_id": source.id,
        "source_group_lineage_id": source.lineage_id,
        "mode": mode,
        "group_index": index,
        "group_count": group_count,
        "span_start": span_start,
        "span_end": span_end,
        "span_seconds": span_end - span_start,
        "segment_duration_seconds": group_total_duration(segments),
        "ipa_chars": sum(segment_ipa_char_count(segment) for segment in segments),
        "merged_text": merged_text,
        "merged_phon": merged_phon,
    }
    name = f"{source.name}_split_{index + 1:04d}"
    return replace(source, name=name, id=group_id, lineage_id=stable_id("lineage", source.lineage_id, group_id), segments=list(segments), metadata=metadata)


def _reachable_end(segments: list[AudioSegment], index: int, max_gap_seconds: float | None) -> int:
    reachable_end = index
    while reachable_end + 1 < len(segments):
        current = segments[reachable_end]
        next_segment = segments[reachable_end + 1]
        gap = next_segment.start - current.end
        same_range = current.start == next_segment.start and current.end == next_segment.end
        if gap < 0 and not same_range:
            break
        if max_gap_seconds is not None and gap > max_gap_seconds:
            break
        reachable_end += 1
    return reachable_end


def _best_group_end(
    segments: list[AudioSegment],
    index: int,
    reachable_end: int,
    min_total_seconds: float,
    min_ipa_chars: int,
    max_ipa_chars: int,
    max_merged_duration_seconds: float | None,
) -> int | None:
    best_end: int | None = None
    for end in range(index + 1, reachable_end + 2):
        candidate = segments[index:end]
        duration = group_total_duration(candidate)
        ipa = sum(segment_ipa_char_count(segment) for segment in candidate)
        if ipa > max_ipa_chars:
            break
        if _span_exceeds_limit(candidate, max_merged_duration_seconds):
            break
        if duration >= min_total_seconds and ipa >= min_ipa_chars:
            best_end = end
    return best_end


def _fallback_group_end(
    segments: list[AudioSegment],
    index: int,
    reachable_end: int,
    max_ipa_chars: int,
    max_merged_duration_seconds: float | None,
) -> int | None:
    fallback_end: int | None = None
    for end in range(index + 1, reachable_end + 2):
        candidate = segments[index:end]
        ipa = sum(segment_ipa_char_count(segment) for segment in candidate)
        if ipa > max_ipa_chars:
            break
        if _span_exceeds_limit(candidate, max_merged_duration_seconds):
            continue
        fallback_end = end
    return fallback_end


def _span_exceeds_limit(segments: list[AudioSegment], max_merged_duration_seconds: float | None) -> bool:
    if max_merged_duration_seconds is None:
        return False
    return group_span_seconds(segments) > float(max_merged_duration_seconds) + 1e-12


def _sample_min_ipa_for_remaining(
    ordered: list[AudioSegment],
    index: int,
    settings: PlanSegmentGroupsSettings,
) -> int:
    if not ordered:
        return settings.min_total_ipa_chars
    remaining = max(0.0, ordered[-1].end - ordered[index].start)
    return sample_min_ipa_chars_from_audio_length(
        audio_length_seconds=remaining,
        min_total_ipa_chars=settings.min_total_ipa_chars,
        max_ipa_chars=settings.max_ipa_chars,
    )
