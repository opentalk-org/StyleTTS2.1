from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from runner.nodes.audio_segments.alignment_merge import AlignmentTrack, alignment_midpoint
from runner.nodes.audio_segments.alignment_merge import project_alignment_to_transcript
from runner.nodes.models import Audio, AudioSegment


DEFAULT_MODEL_PRIORITY = ("src", "canary", "parakeet", "whisper")

DEFAULT_MIN_OVERLAP_RATIO = 0.5

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def speech_segment_records(audio: Audio) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[AudioSegment]] = {}
    for segment in audio.segments:
        key = (round(segment.start * 1000.0), round(segment.end * 1000.0))
        groups.setdefault(key, []).append(segment)
    records: list[dict[str, Any]] = []
    collapsed = 0
    for key in sorted(groups):
        members = groups[key]
        collapsed += len(members) - 1
        records.append(_canonical_record(audio, members))
    return {
        "audio_file_id": str(audio.audio_file_id),
        "segments": records,
        "duplicate_segments_collapsed": collapsed,
    }


def _canonical_record(audio: Audio, members: list[AudioSegment]) -> dict[str, Any]:
    canonical = _select_canonical(members)
    phon = canonical.phon.strip()
    if not phon:
        phon = next((member.phon.strip() for member in members if member.phon.strip()), "")
    text = canonical.text.strip()
    return {
        "source_audio_id": str(audio.audio_file_id),
        "text": text,
        "phon": phon,
        "annotations": canonical.annotations.model_dump(mode="json"),
        "word_count": _word_count(canonical, text),
        "word_times": _word_times(canonical, members),
        "start": float(canonical.start),
        "end": float(canonical.end),
        "duration": float(canonical.duration),
        "model": _segment_model(canonical),
    }


def _word_count(segment: AudioSegment, text: str) -> int:
    # Word-level alignment is the ground truth when present; fall back to whitespace tokens.
    if segment.alignment:
        return len(segment.alignment)
    return len(text.split())


def _word_times(canonical: AudioSegment, members: list[AudioSegment]) -> list[list[float]]:
    # Per-word [start, end] timings used downstream to measure the silence between consecutive
    # words. Prefer the canonical member's alignment; fall back to any member that carries one,
    # mirroring how phon is filled. Empty when no member was force-aligned.
    alignment = canonical.alignment or next((member.alignment for member in members if member.alignment), None)
    if not alignment:
        return []
    return [[float(word["start"]), float(word["end"])] for word in alignment]


def _select_canonical(members: list[AudioSegment]) -> AudioSegment:
    preferred_column = _preferred_column(members)
    if preferred_column is not None:
        for member in members:
            if str(member.metadata.get("text_column", "")) == preferred_column:
                return member
        preferred_model = preferred_column.removeprefix("text_")
        for member in members:
            if _segment_model(member) == preferred_model:
                return member
    ranked = sorted(members, key=lambda member: (_model_rank(member), -len(member.text.strip())))
    return ranked[0]


def _preferred_column(members: list[AudioSegment]) -> str | None:
    for member in members:
        value = member.metadata.get("preferred_text_column")
        if value:
            return str(value)
    return None


def _model_rank(member: AudioSegment) -> int:
    model = _segment_model(member)
    if model in DEFAULT_MODEL_PRIORITY:
        return DEFAULT_MODEL_PRIORITY.index(model)
    return len(DEFAULT_MODEL_PRIORITY)


def _segment_model(member: AudioSegment) -> str:
    for field in ("model", "type_"):
        value = member.metadata.get(field)
        if value:
            return str(value)
    return ""


def deduplicate_overlapping_segments(
    segments: list[AudioSegment],
    *,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> tuple[list[AudioSegment], int]:
    if not segments:
        return [], 0
    kept: list[AudioSegment] = []
    collapsed = 0
    for members in _overlap_clusters(segments, min_overlap_ratio):
        collapsed += len(members) - 1
        kept.append(_consensus_segment(members))
    kept.sort(key=lambda segment: (segment.start, segment.end))
    return kept, collapsed


def _overlap_clusters(segments: list[AudioSegment], min_overlap_ratio: float) -> list[list[AudioSegment]]:
    order = sorted(range(len(segments)), key=lambda index: (segments[index].start, segments[index].end))
    parent = list(range(len(segments)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    active: list[int] = []
    for index in order:
        current = segments[index]
        active = [other for other in active if segments[other].end > current.start]
        for other in active:
            if _overlap_ratio(current, segments[other]) >= min_overlap_ratio:
                union(index, other)
        active.append(index)

    grouped: dict[int, list[AudioSegment]] = {}
    for index in order:
        grouped.setdefault(find(index), []).append(segments[index])
    return list(grouped.values())


def _overlap_ratio(left: AudioSegment, right: AudioSegment) -> float:
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if overlap <= 0.0:
        return 0.0
    shortest = min(left.duration, right.duration)
    if shortest <= 0.0:
        return 1.0
    return overlap / shortest


# Accuracy tilts the vote without replacing member counting.
ACCURACY_WEIGHT_FLOOR = 0.7


def _consensus_segment(members: list[AudioSegment]) -> AudioSegment:
    if len(members) == 1:
        winner = members[0]
        tracks = [AlignmentTrack(winner.alignment or [], preferred=True)]
        alignment, count = project_alignment_to_transcript(
            winner.text, winner.start, winner.end, tracks
        )
        metadata = {**winner.metadata, "alignment_interpolated_words": count}
        return replace(winner, annotations=winner.annotations.model_copy(update={"metadata": metadata}), alignment=alignment)
    normals = [_normalized_text(member) for member in members]
    weights = _accuracy_weights(members)
    ranked = sorted(
        range(len(members)),
        key=lambda index: (
            -_accuracy_weighted_support(normals, weights, index),
            -(members[index].accuracy if members[index].accuracy is not None else 0.0),
            _model_rank(members[index]),
            -len(normals[index]),
        ),
    )
    winner_index = ranked[0]
    winner = members[winner_index]
    # Peer similarity and accuracy drive both disagreement penalty and corroboration bonus.
    others = [index for index in range(len(members)) if index != winner_index]
    sims = [_text_similarity(normals[winner_index], normals[index]) for index in others]
    other_accuracies = [members[index].accuracy for index in others]
    agreement = sum(sims) / len(sims)
    metadata = {
        **winner.metadata,
        "overlap_cluster_size": len(members),
        "overlap_consensus_score": round(agreement, 6),
        "overlap_selected_model_accuracy": winner.accuracy,
        # Keep every member's text + score so nothing is lost when the cluster collapses.
        "overlap_members": [
            {"model": _segment_model(member), "text": member.text.strip(), "accuracy": member.accuracy}
            for member in members
        ],
        "overlap_alternatives": [member.text.strip() for index, member in enumerate(members) if index != winner_index],
    }
    accuracy = _consensus_accuracy(winner.accuracy, sims, other_accuracies)
    other_tracks = [
        [word for word in (member.alignment or []) if winner.start <= alignment_midpoint(word) <= winner.end]
        for index, member in enumerate(members)
        if index != winner_index
    ]
    tracks = [AlignmentTrack(winner.alignment or [], preferred=True)]
    tracks.extend(AlignmentTrack(track) for track in other_tracks)
    alignment, interpolated_words = project_alignment_to_transcript(
        winner.text, winner.start, winner.end, tracks
    )
    metadata["alignment_interpolated_words"] = interpolated_words
    annotations = winner.annotations.model_copy(update={"accuracy": accuracy, "metadata": metadata})
    return replace(winner, annotations=annotations, alignment=alignment)


# Damping keeps correlated model agreement from saturating accuracy.
CONSENSUS_BONUS_DAMPING = 0.5


def _consensus_accuracy(
    winner_accuracy: float | None, sims: list[float], other_accuracies: list[float | None]
) -> float | None:
    """Semi-normalized cluster accuracy: winner score, penalized by disagreement then lifted a
    bounded amount by confident corroboration.
    ``base`` is the winner's acoustic score scaled by mean agreement (model-agnostic, so texts
    not matching always lowers it). On top, each *other* member that both agrees and is confident
    closes part of the remaining gap to 1.0, so a cross-checked segment outranks a lone confident
    one — but the damping keeps consensus from saturating the score. Bounds: ``base <= result < 1``.
    """
    agreement = sum(sims) / len(sims)
    base = agreement if winner_accuracy is None else winner_accuracy * agreement
    corroboration = sum(
        sim * (value if value is not None else 0.0) for sim, value in zip(sims, other_accuracies)
    ) / len(sims)
    return base + (1.0 - base) * corroboration * CONSENSUS_BONUS_DAMPING


def _accuracy_weights(members: list[AudioSegment]) -> list[float]:
    """Per-member vote weights, min-max normalized within the cluster into ``[FLOOR, 1.0]``.

    Normalizing *within the cluster* makes the weight relative (who is most accurate here),
    which neutralizes the fact that engines report accuracy on different scales. Missing
    scores get a neutral weight, and the floor guarantees every member still votes.
    """
    accuracies = [member.accuracy for member in members]
    present = [value for value in accuracies if value is not None]
    if len(present) < 2:
        return [1.0] * len(members)
    low, high = min(present), max(present)
    span = high - low
    if span <= 0.0:
        return [1.0] * len(members)
    weights = []
    for value in accuracies:
        if value is None:
            weights.append(1.0)
        else:
            weights.append(ACCURACY_WEIGHT_FLOOR + (1.0 - ACCURACY_WEIGHT_FLOOR) * (value - low) / span)
    return weights


def _accuracy_weighted_support(normals: list[str], weights: list[float], index: int) -> float:
    """Accuracy-weighted support for a member's text, summed over the whole cluster.

    Self is included (``sim == 1``), which is what makes the score symmetric: two members
    with identical text get identical support regardless of their own weights, so selection
    among them falls through to the accuracy tiebreak instead of favouring whoever happened
    to sit next to the more accurate peers.
    """
    return sum(
        _text_similarity(normals[index], normals[other]) * weights[other]
        for other in range(len(normals))
    )


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _normalized_text(member: AudioSegment) -> str:
    text = (member.text or "").lower().strip()
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
