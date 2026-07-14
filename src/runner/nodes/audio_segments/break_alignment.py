from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable

from runner.nodes.audio_segments.silence_detection import SilenceInterval
from runner.nodes.models import AudioSegment


BREAK_PATTERN = re.compile(r"^<break t=\d+>$")


@dataclass(frozen=True)
class AlignedWord:
    entry_index: int
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class BreakCandidate:
    boundary: int
    interval: SilenceInterval
    duration_ms: int
    overlap: float

    @property
    def tag(self) -> str:
        return f"<break t={self.duration_ms}>"


def annotate_segment(
    segment: AudioSegment,
    silences: list[SilenceInterval],
    min_break_time_ms: int,
    insert_at_start: bool,
    insert_at_end: bool,
    drop_prob: float,
    word_overlap_drop_ratio: float,
    random_value: Callable[[], float] = random.random,
) -> AudioSegment:
    if not segment.alignment:
        return segment
    words = _aligned_words(segment)
    candidates = [
        candidate
        for silence in silences
        if (
            candidate := _select_candidate(
                segment,
                words,
                silence,
                min_break_time_ms,
                insert_at_start,
                insert_at_end,
                word_overlap_drop_ratio,
            )
        ) is not None
        and random_value() >= drop_prob
    ]
    candidates = _without_existing_breaks(segment.alignment, candidates)
    if not candidates:
        return segment
    return replace(
        segment,
        text=_insert_break_text(segment, words, candidates),
        alignment=_insert_break_alignments(segment.alignment, words, candidates),
    )


def _aligned_words(segment: AudioSegment) -> list[AlignedWord]:
    assert segment.alignment is not None, "alignment is required"
    words = []
    previous_start = segment.start
    for index, entry in enumerate(segment.alignment):
        try:
            word = str(entry["word"]).strip()
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid alignment entry for segment {segment.id}: {entry!r}") from error
        valid = (
            word
            and math.isfinite(start)
            and math.isfinite(end)
            and segment.start <= start <= end <= segment.end
            and start >= previous_start
        )
        if not valid:
            raise ValueError(f"invalid alignment timing for segment {segment.id}: {entry!r}")
        previous_start = start
        if not BREAK_PATTERN.fullmatch(word):
            words.append(AlignedWord(index, word, start, end))
    if not words:
        raise ValueError(f"alignment has no words for segment {segment.id}")
    return words


def _select_candidate(
    segment: AudioSegment,
    words: list[AlignedWord],
    silence: SilenceInterval,
    min_break_time_ms: int,
    insert_at_start: bool,
    insert_at_end: bool,
    word_overlap_drop_ratio: float,
) -> BreakCandidate | None:
    interval = SilenceInterval(max(segment.start, silence.start), min(segment.end, silence.end))
    duration = interval.end - interval.start
    if duration <= 0.0:
        return None
    if _word_overlap_duration(interval, words) / duration > word_overlap_drop_ratio:
        return None
    duration_ms = int(round(duration * 1000.0))
    if duration_ms < min_break_time_ms:
        return None
    boundaries = []
    if insert_at_start:
        boundaries.append((0, segment.start, words[0].start))
    boundaries.extend(
        (index, previous.end, following.start)
        for index, (previous, following) in enumerate(zip(words, words[1:], strict=False), start=1)
    )
    if insert_at_end:
        boundaries.append((len(words), words[-1].end, segment.end))
    eligible = [
        BreakCandidate(boundary, interval, duration_ms, overlap)
        for boundary, start, end in boundaries
        if (overlap := _overlap(interval.start, interval.end, start, end)) > 0.0
    ]
    return min(eligible, key=lambda candidate: (-candidate.overlap, candidate.boundary)) if eligible else None


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _word_overlap_duration(interval: SilenceInterval, words: list[AlignedWord]) -> float:
    intersections = sorted(
        (max(interval.start, word.start), min(interval.end, word.end))
        for word in words
        if _overlap(interval.start, interval.end, word.start, word.end) > 0.0
    )
    if not intersections:
        return 0.0
    total = 0.0
    current_start, current_end = intersections[0]
    for start, end in intersections[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _without_existing_breaks(
    alignment: list[dict],
    candidates: list[BreakCandidate],
) -> list[BreakCandidate]:
    existing = {
        (str(entry["word"]), round(float(entry["start"]), 9), round(float(entry["end"]), 9))
        for entry in alignment
        if BREAK_PATTERN.fullmatch(str(entry["word"]).strip())
    }
    return [
        candidate
        for candidate in candidates
        if (candidate.tag, round(candidate.interval.start, 9), round(candidate.interval.end, 9)) not in existing
    ]


def _insert_break_text(
    segment: AudioSegment,
    words: list[AlignedWord],
    candidates: list[BreakCandidate],
) -> str:
    word_positions = _word_positions(segment, words)
    boundaries: dict[int, list[BreakCandidate]] = defaultdict(list)
    for candidate in candidates:
        boundaries[candidate.boundary].append(candidate)
    text = segment.text
    insertions = []
    for boundary, breaks in boundaries.items():
        position = 0 if boundary == 0 else len(text) if boundary == len(words) else word_positions[boundary][0]
        tags = " ".join(item.tag for item in sorted(breaks, key=lambda item: item.interval.start))
        insertions.append((position, tags))
    for position, tags in sorted(insertions, reverse=True):
        left = text[:position].rstrip()
        right = text[position:].lstrip()
        text = " ".join(part for part in (left, tags, right) if part)
    return text


def _word_positions(segment: AudioSegment, words: list[AlignedWord]) -> list[tuple[int, int]]:
    positions = []
    cursor = 0
    for word in words:
        start = segment.text.find(word.word, cursor)
        if start < 0:
            raise ValueError(f"alignment word {word.word!r} not found in transcript for segment {segment.id}")
        end = start + len(word.word)
        positions.append((start, end))
        cursor = end
    return positions


def _insert_break_alignments(
    alignment: list[dict],
    words: list[AlignedWord],
    candidates: list[BreakCandidate],
) -> list[dict]:
    by_index: dict[int, list[BreakCandidate]] = defaultdict(list)
    for candidate in candidates:
        insertion_index = words[candidate.boundary].entry_index if candidate.boundary < len(words) else len(alignment)
        by_index[insertion_index].append(candidate)
    updated = []
    for index, entry in enumerate(alignment):
        updated.extend(_break_entries(by_index[index]))
        updated.append(entry)
    updated.extend(_break_entries(by_index[len(alignment)]))
    return updated


def _break_entries(candidates: list[BreakCandidate]) -> list[dict[str, float | str]]:
    return [
        {"word": candidate.tag, "start": candidate.interval.start, "end": candidate.interval.end}
        for candidate in sorted(candidates, key=lambda item: item.interval.start)
    ]
