from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import log

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.sparse import coo_matrix
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import expected_mutual_information

from runner.nodes.speaker_clustering.audit_metrics.distributions import (
    ScoreDistribution,
    ScoreSample as ScoreSample,
    score_distribution as score_distribution,
)


@dataclass(frozen=True)
class AssignmentAuditRow:
    segment_id: str
    cluster_id: int | None
    true_label: str | None
    centroid_score: float | None
    second_score: float | None


class LabeledAuditMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    labeled_count: int = Field(ge=0)
    clustered_labeled_count: int = Field(ge=0)
    pair_precision: float | None = Field(ge=0.0, le=1.0)
    pair_recall: float | None = Field(ge=0.0, le=1.0)
    weighted_purity: float | None = Field(ge=0.0, le=1.0)
    adjusted_rand_index: float | None = Field(ge=-1.0, le=1.0)
    adjusted_mutual_info: float | None = Field(ge=-1.0, le=1.0)
    fragmented_speaker_count: int = Field(ge=0)
    max_clusters_per_true_speaker: int = Field(ge=0)
    max_true_speakers_in_cluster: int = Field(ge=0)
    suspicious_cluster_ids: tuple[int, ...]


class SpeakerAuditMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    labeled: LabeledAuditMetrics
    centroid_scores: ScoreDistribution
    second_scores: ScoreDistribution
    margins: ScoreDistribution
    pair_scores: ScoreDistribution


def compute_labeled_metrics(
    rows: Iterable[AssignmentAuditRow],
) -> LabeledAuditMetrics:
    cells: Counter[tuple[int, str]] = Counter()
    cluster_totals: Counter[int] = Counter()
    clustered_truth_totals: Counter[str] = Counter()
    all_truth_totals: Counter[str] = Counter()
    labeled_count = 0
    clustered_count = 0
    for row in rows:
        if row.true_label is None:
            continue
        labeled_count += 1
        all_truth_totals[row.true_label] += 1
        if row.cluster_id is None:
            continue
        clustered_count += 1
        cells[(row.cluster_id, row.true_label)] += 1
        cluster_totals[row.cluster_id] += 1
        clustered_truth_totals[row.true_label] += 1
    predicted_pairs = sum(_pairs(count) for count in cluster_totals.values())
    correct_pairs = sum(_pairs(count) for count in cells.values())
    true_pairs = sum(_pairs(count) for count in all_truth_totals.values())
    cluster_label_counts, truth_cluster_counts = _distinct_counts(cells)
    return LabeledAuditMetrics(
        labeled_count=labeled_count,
        clustered_labeled_count=clustered_count,
        pair_precision=_ratio(correct_pairs, predicted_pairs),
        pair_recall=_ratio(correct_pairs, true_pairs),
        weighted_purity=_weighted_purity(cells, cluster_totals),
        adjusted_rand_index=_adjusted_rand(cells, cluster_totals, clustered_truth_totals),
        adjusted_mutual_info=_adjusted_mutual_info(
            cells, cluster_totals, clustered_truth_totals
        ),
        fragmented_speaker_count=sum(
            count > 1 for count in truth_cluster_counts.values()
        ),
        max_clusters_per_true_speaker=max(truth_cluster_counts.values(), default=0),
        max_true_speakers_in_cluster=max(cluster_label_counts.values(), default=0),
        suspicious_cluster_ids=tuple(
            sorted(cluster for cluster, count in cluster_label_counts.items() if count > 1)
        ),
    )


def _distinct_counts(
    cells: Counter[tuple[int, str]],
) -> tuple[Counter[int], Counter[str]]:
    clusters: Counter[int] = Counter()
    truths: Counter[str] = Counter()
    for cluster_id, true_label in cells:
        clusters[cluster_id] += 1
        truths[true_label] += 1
    return clusters, truths


def _weighted_purity(
    cells: Counter[tuple[int, str]], cluster_totals: Counter[int]
) -> float | None:
    if not cluster_totals:
        return None
    maxima: Counter[int] = Counter()
    for (cluster_id, _label), count in cells.items():
        maxima[cluster_id] = max(maxima[cluster_id], count)
    return sum(maxima.values()) / sum(cluster_totals.values())


def _adjusted_rand(
    cells: Counter[tuple[int, str]],
    cluster_totals: Counter[int],
    truth_totals: Counter[str],
) -> float | None:
    count = sum(cluster_totals.values())
    if count < 2:
        return None
    total_pairs = _pairs(count)
    index = sum(_pairs(value) for value in cells.values())
    cluster_index = sum(_pairs(value) for value in cluster_totals.values())
    truth_index = sum(_pairs(value) for value in truth_totals.values())
    expected = cluster_index * truth_index / total_pairs
    denominator = (cluster_index + truth_index) / 2 - expected
    return 1.0 if denominator == 0.0 else (index - expected) / denominator


def _adjusted_mutual_info(
    cells: Counter[tuple[int, str]],
    cluster_totals: Counter[int],
    truth_totals: Counter[str],
) -> float | None:
    count = sum(cluster_totals.values())
    if count < 2:
        return None
    cluster_ids = {value: index for index, value in enumerate(sorted(cluster_totals))}
    labels = {value: index for index, value in enumerate(sorted(truth_totals))}
    row_ids = [cluster_ids[cluster] for cluster, _label in cells]
    column_ids = [labels[label] for _cluster, label in cells]
    values = list(cells.values())
    contingency = coo_matrix(
        (values, (row_ids, column_ids)),
        shape=(len(cluster_ids), len(labels)),
        dtype=np.int64,
    ).tocsr()
    mutual_info = mutual_info_score(None, None, contingency=contingency)
    expected = expected_mutual_information(contingency, count)
    denominator = (
        _entropy(cluster_totals.values()) + _entropy(truth_totals.values())
    ) / 2 - expected
    return 1.0 if denominator == 0.0 else (mutual_info - expected) / denominator


def _entropy(counts: Iterable[int]) -> float:
    values = list(counts)
    total = sum(values)
    return -sum((value / total) * log(value / total) for value in values if value)


def _pairs(count: int) -> int:
    return count * (count - 1) // 2


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator

