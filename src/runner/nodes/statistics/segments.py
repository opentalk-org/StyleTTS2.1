from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

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
    return {
        "source_audio_id": str(audio.audio_file_id),
        "text": canonical.text.strip(),
        "phon": phon,
        "speaker": (canonical.speaker or "").strip(),
        "start": float(canonical.start),
        "end": float(canonical.end),
        "duration": float(canonical.duration),
        "model": _segment_model(canonical),
    }


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


# Least-confident cluster member still keeps this fraction of a vote, so no segment is
# silently dropped from the consensus and a numeric majority is not overturned by a single
# very confident model — confidence tilts the vote, it does not replace counting.
CONFIDENCE_WEIGHT_FLOOR = 0.7


def _consensus_segment(members: list[AudioSegment]) -> AudioSegment:
    if len(members) == 1:
        return members[0]
    normals = [_normalized_text(member) for member in members]
    weights = _confidence_weights(members)
    ranked = sorted(
        range(len(members)),
        key=lambda index: (
            -_confidence_weighted_support(normals, weights, index),
            -(members[index].confidence if members[index].confidence is not None else 0.0),
            _model_rank(members[index]),
            -len(normals[index]),
        ),
    )
    winner_index = ranked[0]
    winner = members[winner_index]
    # Similarity + confidence of every other member relative to the winner, used for both the
    # disagreement penalty and the corroboration bonus.
    others = [index for index in range(len(members)) if index != winner_index]
    sims = [_text_similarity(normals[winner_index], normals[index]) for index in others]
    other_confidences = [members[index].confidence for index in others]
    # Mean text similarity of the winner to the rest of the cluster: model-agnostic, in [0, 1].
    agreement = sum(sims) / len(sims)
    metadata = {
        **winner.metadata,
        "overlap_cluster_size": len(members),
        "overlap_consensus_score": round(agreement, 6),
        "overlap_selected_model_confidence": winner.confidence,
        # Keep every member's text + score so nothing is lost when the cluster collapses.
        "overlap_members": [
            {"model": _segment_model(member), "text": member.text.strip(), "confidence": member.confidence}
            for member in members
        ],
        "overlap_alternatives": [member.text.strip() for index, member in enumerate(members) if index != winner_index],
    }
    confidence = _consensus_confidence(winner.confidence, sims, other_confidences)
    return replace(winner, confidence=confidence, metadata=metadata)


# How much a fully-confident, fully-agreeing peer can lift the score toward 1.0. Damped well
# below 1.0 because engine confidences are not calibrated and models make correlated errors,
# so agreement is corroborating evidence, not independent probability to be multiplied out.
CONSENSUS_BONUS_DAMPING = 0.5


def _consensus_confidence(
    winner_confidence: float | None, sims: list[float], other_confidences: list[float | None]
) -> float | None:
    """Semi-normalized cluster confidence: winner score, penalized by disagreement then lifted a
    bounded amount by confident corroboration.

    ``base`` is the winner's acoustic score scaled by mean agreement (model-agnostic, so texts
    not matching always lowers it). On top, each *other* member that both agrees and is confident
    closes part of the remaining gap to 1.0, so a cross-checked segment outranks a lone confident
    one — but the damping keeps consensus from saturating the score. Bounds: ``base <= result < 1``.
    """
    agreement = sum(sims) / len(sims)
    base = agreement if winner_confidence is None else winner_confidence * agreement
    corroboration = sum(
        sim * (conf if conf is not None else 0.0) for sim, conf in zip(sims, other_confidences)
    ) / len(sims)
    return base + (1.0 - base) * corroboration * CONSENSUS_BONUS_DAMPING


def _confidence_weights(members: list[AudioSegment]) -> list[float]:
    """Per-member vote weights, min-max normalized within the cluster into ``[FLOOR, 1.0]``.

    Normalizing *within the cluster* makes the weight relative (who is most confident here),
    which neutralizes the fact that engines report confidence on different scales. Missing
    scores get a neutral weight, and the floor guarantees every member still votes.
    """
    confidences = [member.confidence for member in members]
    present = [value for value in confidences if value is not None]
    if len(present) < 2:
        return [1.0] * len(members)
    low, high = min(present), max(present)
    span = high - low
    if span <= 0.0:
        return [1.0] * len(members)
    weights = []
    for value in confidences:
        if value is None:
            weights.append(1.0)
        else:
            weights.append(CONFIDENCE_WEIGHT_FLOOR + (1.0 - CONFIDENCE_WEIGHT_FLOOR) * (value - low) / span)
    return weights


def _confidence_weighted_support(normals: list[str], weights: list[float], index: int) -> float:
    """Confidence-weighted support for a member's text, summed over the whole cluster.

    Self is included (``sim == 1``), which is what makes the score symmetric: two members
    with identical text get identical support regardless of their own weights, so selection
    among them falls through to the confidence tiebreak instead of favouring whoever happened
    to sit next to the more confident peers.
    """
    return sum(
        _text_similarity(normals[index], normals[other]) * weights[other]
        for other in range(len(normals))
    )


def _agreement_score(normals: list[str], index: int) -> float:
    return sum(_text_similarity(normals[index], normals[other]) for other in range(len(normals)) if other != index)


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
