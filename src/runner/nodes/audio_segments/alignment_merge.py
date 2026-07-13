from __future__ import annotations

from typing import Any


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
