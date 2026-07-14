from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapreplace, heappush
from typing import TypeAlias
from collections.abc import Iterable

from runner.nodes.speaker_clustering.audit_report.models import (
    AssignmentAuditInput,
    ListeningEntry,
    ListeningManifest,
)


Rank: TypeAlias = tuple[int | float, str, str, str]


@dataclass(frozen=True)
class _RankedEntry:
    rank: Rank
    entry: ListeningEntry = field(compare=False)

    def __lt__(self, other: _RankedEntry) -> bool:
        return self.rank > other.rank


class _BoundedSelection:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._heap: list[_RankedEntry] = []

    def add(self, rank: Rank, entry: ListeningEntry) -> None:
        candidate = _RankedEntry(rank=rank, entry=entry)
        if len(self._heap) < self._limit:
            heappush(self._heap, candidate)
        elif rank < self._heap[0].rank:
            heapreplace(self._heap, candidate)

    def ordered(self) -> tuple[ListeningEntry, ...]:
        return tuple(
            item.entry for item in sorted(self._heap, key=lambda item: item.rank)
        )


def select_listening_manifest(
    rows: Iterable[AssignmentAuditInput],
    suspicious_cluster_ids: frozenset[int],
    limit: int,
) -> ListeningManifest:
    if limit <= 0:
        raise ValueError("category_limit must be positive")
    worst = _BoundedSelection(limit)
    cross_cluster = _BoundedSelection(limit)
    boundaries = _BoundedSelection(limit)
    suspicious = _BoundedSelection(limit)
    for row in rows:
        entry = _listening_entry(row)
        tie = (row.segment_id, row.audio_id, entry.model_dump_json())
        if row.cluster_id is not None and row.best_score is not None:
            worst.add((row.best_score, *tie), entry)
        if row.second_cluster_id is not None and row.second_score is not None:
            cross_cluster.add((-row.second_score, *tie), entry)
        if row.margin is not None:
            boundaries.add((row.margin, *tie), entry)
        if row.cluster_id in suspicious_cluster_ids and row.true_label is not None:
            suspicious.add(
                (row.cluster_id, row.true_label, row.segment_id, tie[2]), entry
            )
    return ListeningManifest(
        worst_within_cluster=worst.ordered(),
        closest_cross_cluster=cross_cluster.ordered(),
        low_margin_boundaries=boundaries.ordered(),
        suspicious_labeled_merges=suspicious.ordered(),
    )


def _listening_entry(row: AssignmentAuditInput) -> ListeningEntry:
    return ListeningEntry(
        segment_id=row.segment_id,
        audio_id=row.audio_id,
        duration_seconds=row.duration_seconds,
        cluster_id=row.cluster_id,
        best_cluster_id=row.best_cluster_id,
        second_cluster_id=row.second_cluster_id,
        true_label=row.true_label,
        best_score=row.best_score,
        second_score=row.second_score,
        margin=row.margin,
    )
