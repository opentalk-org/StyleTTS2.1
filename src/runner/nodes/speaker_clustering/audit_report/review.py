from __future__ import annotations

from collections.abc import Mapping

from runner.nodes.speaker_clustering.audit_report.models import (
    AssignmentAuditDocument,
    ListeningEntry,
)
from shared.db.reviews.schemas import (
    AudioSegmentReviewMedia,
    ReviewField,
    ReviewGroup,
    ReviewItem,
    ReviewMetric,
    ReviewPayload,
    ReviewTone,
)
from shared.db.speakers.schemas import ClusteringOutcomeCounts


MediaKey = tuple[str, str]


def speaker_review_payload(
    document: AssignmentAuditDocument,
    media_by_segment: Mapping[MediaKey, AudioSegmentReviewMedia],
    outcomes: ClusteringOutcomeCounts,
) -> ReviewPayload:
    labeled = document.metrics.labeled
    warnings = _warnings(
        labeled.max_true_speakers_in_cluster, labeled.fragmented_speaker_count
    )
    manifest = document.listening_manifest
    groups = (
        _group(
            "weakest_within_cluster",
            "Weakest accepted members",
            "Lowest similarity to the assigned cluster prototype.",
            "warning",
            manifest.worst_within_cluster,
            media_by_segment,
        ),
        _group(
            "closest_cross_cluster",
            "Closest cross-cluster candidates",
            "Segments whose second-best cluster is unusually similar.",
            "warning",
            manifest.closest_cross_cluster,
            media_by_segment,
        ),
        _group(
            "low_margin_boundaries",
            "Lowest assignment margins",
            "Assignments with the smallest best-versus-second score margin.",
            "warning",
            manifest.low_margin_boundaries,
            media_by_segment,
        ),
        _group(
            "suspicious_labeled_merges",
            "Suspicious labeled merges",
            "Clusters containing more than one known speaker label.",
            "danger",
            manifest.suspicious_labeled_merges,
            media_by_segment,
        ),
    )
    return ReviewPayload(
        metrics=_metrics(document, outcomes),
        warnings=warnings,
        groups=groups,
    )


def _metrics(
    document: AssignmentAuditDocument, outcomes: ClusteringOutcomeCounts
) -> tuple[ReviewMetric, ...]:
    labeled = document.metrics.labeled
    total = document.total_rows
    coverage = 0.0 if total == 0 else outcomes.accepted / total
    return (
        _quality_metric("pair_precision", "Pair precision", labeled.pair_precision),
        _quality_metric("weighted_purity", "Weighted purity", labeled.weighted_purity),
        _quality_metric("pair_recall", "Pair recall", labeled.pair_recall),
        _quality_metric(
            "adjusted_rand_index", "Adjusted Rand", labeled.adjusted_rand_index
        ),
        _quality_metric(
            "adjusted_mutual_info", "Adjusted mutual info", labeled.adjusted_mutual_info
        ),
        ReviewMetric(
            key="accepted_coverage",
            label="Accepted coverage",
            value=_percent(coverage),
            numeric_value=coverage,
            tone="neutral",
        ),
        ReviewMetric(
            key="accepted_count",
            label="Accepted segments",
            value=f"{outcomes.accepted:,}",
            numeric_value=float(outcomes.accepted),
            tone="neutral",
        ),
        ReviewMetric(
            key="ambiguous_count",
            label="Ambiguous segments",
            value=f"{outcomes.ambiguous:,}",
            numeric_value=float(outcomes.ambiguous),
            tone="warning" if outcomes.ambiguous else "success",
        ),
    )


def _quality_metric(key: str, label: str, value: float | None) -> ReviewMetric:
    tone: ReviewTone
    if value is None:
        tone = "neutral"
    elif value >= 0.995:
        tone = "success"
    elif value >= 0.98:
        tone = "warning"
    else:
        tone = "danger"
    return ReviewMetric(
        key=key,
        label=label,
        value="Not labeled" if value is None else _percent(value),
        numeric_value=value,
        tone=tone,
    )


def _group(
    key: str,
    title: str,
    explanation: str,
    tone: ReviewTone,
    entries: tuple[ListeningEntry, ...],
    media_by_segment: Mapping[MediaKey, AudioSegmentReviewMedia],
) -> ReviewGroup:
    return ReviewGroup(
        key=key,
        title=title,
        explanation=explanation,
        tone=tone,
        items=tuple(_item(entry, media_by_segment) for entry in entries),
    )


def _item(
    entry: ListeningEntry,
    media_by_segment: Mapping[MediaKey, AudioSegmentReviewMedia],
) -> ReviewItem:
    media = media_by_segment[(entry.audio_id, entry.segment_id)]
    fields = (
        ReviewField(key="cluster", label="Cluster", value=_optional(entry.cluster_id)),
        ReviewField(
            key="label", label="Known speaker", value=entry.true_label or "Unknown"
        ),
        ReviewField(
            key="best_score", label="Best score", value=_score(entry.best_score)
        ),
        ReviewField(
            key="second_score", label="Second score", value=_score(entry.second_score)
        ),
        ReviewField(key="margin", label="Margin", value=_score(entry.margin)),
    )
    return ReviewItem(
        key=f"{entry.audio_id}:{entry.segment_id}",
        title=media.name,
        subtitle=entry.segment_id,
        fields=fields,
        media=(media,),
    )


def _warnings(max_speakers: int, fragmented: int) -> tuple[str, ...]:
    values = []
    if max_speakers > 1:
        values.append(f"{max_speakers - 1} cluster contains multiple labeled speakers")
    if fragmented:
        noun = "speaker is" if fragmented == 1 else "speakers are"
        values.append(f"{fragmented} known {noun} fragmented across clusters")
    return tuple(values)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _optional(value: object | None) -> str:
    return "—" if value is None else str(value)
