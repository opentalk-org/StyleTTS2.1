from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentTrack:
    words: list[dict[str, Any]]
    preferred: bool = False


@dataclass(frozen=True)
class _AlignmentCandidate:
    entry: dict[str, Any]
    preferred: bool
    track_index: int


def merge_alignment_tracks(
    tracks: list[list[dict[str, Any]]],
    preferred_window_sec: float,
) -> list[dict[str, Any]] | None:
    merged: list[dict[str, Any]] = []
    for track in tracks:
        incoming = [dict(word) for word in track]
        matches = _match_words(merged, incoming, preferred_window_sec)
        matched_incoming = {incoming_index for _, incoming_index in matches}
        for merged_index, incoming_index in matches:
            if _score(incoming[incoming_index]) > _score(merged[merged_index]):
                merged[merged_index] = incoming[incoming_index]
        known_words = {_normalized(word["word"]) for word in merged}
        merged.extend(
            word
            for index, word in enumerate(incoming)
            if index not in matched_incoming and _normalized(word["word"]) not in known_words
        )
    merged.sort(key=lambda word: float(word["start"]))
    return merged or None


def alignment_midpoint(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2


def project_alignment_to_transcript(
    text: str,
    segment_start: float,
    segment_end: float,
    tracks: list[AlignmentTrack],
) -> tuple[list[dict[str, Any]] | None, int]:
    tokens = text.strip().split()
    if not tokens:
        return None, 0
    candidates: list[list[_AlignmentCandidate]] = [[] for _ in tokens]
    for track_index, track in enumerate(tracks):
        for token_index, word_index in _sequence_mapping(tokens, track.words):
            candidates[token_index].append(
                _AlignmentCandidate(track.words[word_index], track.preferred, track_index)
            )
    selected = _select_timings(tokens, candidates, segment_start, segment_end)
    interpolated = _interpolate_missing(tokens, selected, segment_start, segment_end)
    _validate_projection(tokens, selected, segment_start, segment_end)
    return [entry for entry in selected if entry is not None], interpolated


def _sequence_mapping(tokens: list[str], words: list[dict[str, Any]]) -> list[tuple[int, int]]:
    token_keys = [_normalized(token) for token in tokens]
    word_keys = [_normalized(word["word"]) for word in words]
    table = [[0] * (len(words) + 1) for _ in range(len(tokens) + 1)]
    for token_index in range(len(tokens) - 1, -1, -1):
        for word_index in range(len(words) - 1, -1, -1):
            matched = (
                1 + table[token_index + 1][word_index + 1]
                if token_keys[token_index] and token_keys[token_index] == word_keys[word_index]
                else 0
            )
            table[token_index][word_index] = max(
                matched,
                table[token_index + 1][word_index],
                table[token_index][word_index + 1],
            )
    mapping = []
    token_index = word_index = 0
    while token_index < len(tokens) and word_index < len(words):
        matched = token_keys[token_index] and token_keys[token_index] == word_keys[word_index]
        if matched and table[token_index][word_index] == 1 + table[token_index + 1][word_index + 1]:
            mapping.append((token_index, word_index))
            token_index += 1
            word_index += 1
        elif table[token_index + 1][word_index] >= table[token_index][word_index + 1]:
            token_index += 1
        else:
            word_index += 1
    return mapping


def _select_timings(
    tokens: list[str],
    candidates: list[list[_AlignmentCandidate]],
    segment_start: float,
    segment_end: float,
) -> list[dict[str, Any] | None]:
    selected: list[dict[str, Any] | None] = []
    previous_start = segment_start
    for token, options in zip(tokens, candidates, strict=True):
        entry = None
        for candidate in sorted(options, key=_candidate_rank):
            timed = _bounded_entry(candidate.entry, token, segment_start, segment_end)
            if timed is not None and float(timed["start"]) >= previous_start:
                entry = timed
                previous_start = float(timed["start"])
                break
        selected.append(entry)
    return selected


def _candidate_rank(candidate: _AlignmentCandidate) -> tuple[float, bool, int]:
    raw_score = candidate.entry.get("score")
    score = float(raw_score) if raw_score is not None else float("-inf")
    return -score, not candidate.preferred, candidate.track_index


def _bounded_entry(
    entry: dict[str, Any],
    token: str,
    segment_start: float,
    segment_end: float,
) -> dict[str, Any] | None:
    start = float(entry["start"])
    end = float(entry["end"])
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        return None
    bounded = dict(entry)
    bounded.update(
        word=token,
        start=max(segment_start, min(segment_end, start)),
        end=max(segment_start, min(segment_end, end)),
    )
    return bounded if float(bounded["start"]) <= float(bounded["end"]) else None


def _interpolate_missing(
    tokens: list[str],
    entries: list[dict[str, Any] | None],
    segment_start: float,
    segment_end: float,
) -> int:
    interpolated = 0
    index = 0
    while index < len(entries):
        if entries[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(entries) and entries[index] is None:
            index += 1
        left = segment_start if run_start == 0 else float(entries[run_start - 1]["end"])
        right = segment_end if index == len(entries) else float(entries[index]["start"])
        _fill_run(tokens, entries, run_start, index, left, right, segment_start, segment_end)
        interpolated += index - run_start
    return interpolated


def _fill_run(
    tokens: list[str],
    entries: list[dict[str, Any] | None],
    run_start: int,
    run_end: int,
    left: float,
    right: float,
    segment_start: float,
    segment_end: float,
) -> None:
    count = run_end - run_start
    if right >= left:
        step = (right - left) / count
        for offset, token_index in enumerate(range(run_start, run_end)):
            entries[token_index] = {
                "word": tokens[token_index],
                "start": left + offset * step,
                "end": left + (offset + 1) * step,
                "interpolated": True,
            }
        return
    previous = entries[run_start - 1]
    following = entries[run_end]
    previous_midpoint = alignment_midpoint(previous) if previous is not None else segment_start
    following_midpoint = alignment_midpoint(following) if following is not None else segment_end
    lower = float(previous["start"]) if previous is not None else segment_start
    upper = float(following["start"]) if following is not None else segment_end
    for offset, token_index in enumerate(range(run_start, run_end), start=1):
        point = previous_midpoint + (following_midpoint - previous_midpoint) * offset / (count + 1)
        point = max(lower, min(upper, point))
        entries[token_index] = {
            "word": tokens[token_index],
            "start": point,
            "end": point,
            "interpolated": True,
        }


def _validate_projection(
    tokens: list[str],
    entries: list[dict[str, Any] | None],
    segment_start: float,
    segment_end: float,
) -> None:
    if any(entry is None for entry in entries):
        raise ValueError("transcript alignment projection left unmatched entries")
    complete = [entry for entry in entries if entry is not None]
    if [str(entry["word"]) for entry in complete] != tokens:
        raise ValueError("transcript alignment projection changed transcript tokens")
    previous_start = segment_start
    for entry in complete:
        start, end = float(entry["start"]), float(entry["end"])
        valid = (
            math.isfinite(start)
            and math.isfinite(end)
            and segment_start <= start <= end <= segment_end
            and start >= previous_start
        )
        if not valid:
            raise ValueError(f"invalid projected alignment entry: {entry!r}")
        previous_start = start


def _match_words(
    merged: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    preferred_window_sec: float,
) -> list[tuple[int, int]]:
    candidates = []
    for merged_index, kept in enumerate(merged):
        for incoming_index, word in enumerate(incoming):
            if _normalized(kept["word"]) == _normalized(word["word"]):
                distance = abs(alignment_midpoint(kept) - alignment_midpoint(word))
                candidates.append((distance > preferred_window_sec, distance, merged_index, incoming_index))

    matches = []
    used_merged: set[int] = set()
    used_incoming: set[int] = set()
    for _, _, merged_index, incoming_index in sorted(candidates):
        if merged_index in used_merged or incoming_index in used_incoming:
            continue
        used_merged.add(merged_index)
        used_incoming.add(incoming_index)
        matches.append((merged_index, incoming_index))
    return matches


def _normalized(word: str) -> str:
    return "".join(char for char in str(word).lower() if char.isalnum())


def _score(word: dict[str, Any]) -> float:
    score = word.get("score")
    return float(score) if score is not None else -1.0
