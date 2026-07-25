from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from runner.nodes.speaker_clustering.shards import EmbeddingQuality
from shared.db.speakers.schemas import SpeakerAssignmentOutcome


class AssignmentReason(StrEnum):
    ACCEPTED = "accepted"
    QUALITY_REJECTED = "quality_rejected"
    SUSPICIOUS_CLUSTER = "suspicious_cluster"
    NO_ESTABLISHED_CLUSTER = "no_established_cluster"
    BELOW_NEW_THRESHOLD = "below_new_threshold"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    BELOW_ACCEPT_THRESHOLD = "below_accept_threshold"


class AssignmentPolicy(BaseModel):
    accept_threshold: float = Field(ge=-1.0, le=1.0)
    min_margin: float = Field(ge=0.0, le=2.0)
    new_threshold: float = Field(ge=-1.0, le=1.0)
    dispersion_penalty: float = Field(ge=0.0)
    threshold_version: str = Field(min_length=1)

@dataclass(frozen=True)
class CandidateScores:
    cluster_ids: list[int]
    scores: list[float]
    best_dispersion: float
    best_suspicious: bool

@dataclass(frozen=True)
class AssignmentDecision:
    outcome: SpeakerAssignmentOutcome
    cluster_id: int | None
    best_cluster_id: int | None
    second_cluster_id: int | None
    best_score: float | None
    second_score: float | None
    margin: float | None
    candidate_cluster_ids: list[int]
    candidate_scores: list[float]
    threshold_version: str
    reason: AssignmentReason
    rejection_reason: str | None


def decide(
    quality: EmbeddingQuality,
    rejection_reason: str | None,
    provisional_cluster_id: int,
    provisional_cluster_suspicious: bool,
    candidates: CandidateScores | None,
    policy: AssignmentPolicy,
) -> AssignmentDecision:
    if quality is EmbeddingQuality.REJECTED:
        if rejection_reason is None:
            raise ValueError("rejected embedding requires a rejection reason")
        return _decision(
            SpeakerAssignmentOutcome.REJECTED,
            None,
            None,
            policy,
            AssignmentReason.QUALITY_REJECTED,
            rejection_reason,
        )
    if provisional_cluster_suspicious:
        return _decision(
            SpeakerAssignmentOutcome.AMBIGUOUS,
            None,
            candidates,
            policy,
            AssignmentReason.SUSPICIOUS_CLUSTER,
            None,
        )
    if candidates is None:
        return _decision(
            SpeakerAssignmentOutcome.PROVISIONAL_NEW,
            provisional_cluster_id,
            None,
            policy,
            AssignmentReason.NO_ESTABLISHED_CLUSTER,
            None,
        )
    best_score = candidates.scores[0]
    second_score = candidates.scores[1] if len(candidates.scores) > 1 else None
    margin = None if second_score is None else best_score - second_score
    adjusted_threshold = min(
        1.0,
        policy.accept_threshold
        + candidates.best_dispersion * policy.dispersion_penalty,
    )
    if candidates.best_suspicious:
        outcome = SpeakerAssignmentOutcome.AMBIGUOUS
        cluster_id = None
        reason = AssignmentReason.SUSPICIOUS_CLUSTER
    elif best_score >= adjusted_threshold and (
        margin is None or margin >= policy.min_margin
    ):
        outcome = SpeakerAssignmentOutcome.ACCEPTED
        cluster_id = candidates.cluster_ids[0]
        reason = AssignmentReason.ACCEPTED
    elif best_score <= policy.new_threshold:
        outcome = SpeakerAssignmentOutcome.PROVISIONAL_NEW
        cluster_id = provisional_cluster_id
        reason = AssignmentReason.BELOW_NEW_THRESHOLD
    elif margin is not None and margin < policy.min_margin:
        outcome = SpeakerAssignmentOutcome.AMBIGUOUS
        cluster_id = None
        reason = AssignmentReason.INSUFFICIENT_MARGIN
    else:
        outcome = SpeakerAssignmentOutcome.AMBIGUOUS
        cluster_id = None
        reason = AssignmentReason.BELOW_ACCEPT_THRESHOLD
    return AssignmentDecision(
        outcome=outcome,
        cluster_id=cluster_id,
        best_cluster_id=candidates.cluster_ids[0],
        second_cluster_id=(
            candidates.cluster_ids[1] if len(candidates.cluster_ids) > 1 else None
        ),
        best_score=best_score,
        second_score=second_score,
        margin=margin,
        candidate_cluster_ids=candidates.cluster_ids,
        candidate_scores=candidates.scores,
        threshold_version=policy.threshold_version,
        reason=reason,
        rejection_reason=None,
    )


def _decision(
    outcome: SpeakerAssignmentOutcome,
    cluster_id: int | None,
    candidates: CandidateScores | None,
    policy: AssignmentPolicy,
    reason: AssignmentReason,
    rejection_reason: str | None,
) -> AssignmentDecision:
    cluster_ids = [] if candidates is None else candidates.cluster_ids
    scores = [] if candidates is None else candidates.scores
    return AssignmentDecision(
        outcome=outcome,
        cluster_id=cluster_id,
        best_cluster_id=None,
        second_cluster_id=None,
        best_score=None,
        second_score=None,
        margin=None,
        candidate_cluster_ids=cluster_ids,
        candidate_scores=scores,
        threshold_version=policy.threshold_version,
        reason=reason,
        rejection_reason=rejection_reason,
    )
