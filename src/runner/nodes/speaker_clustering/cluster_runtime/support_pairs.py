from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from runner.nodes.speaker_clustering.edge_shards import EdgeBlock


def consolidate_labels(
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    min_support_pairs: int,
    max_members: int,
    *,
    block_rows: int,
    prototype_neighbors: np.ndarray,
    check_cancel: Callable[[], None] | None = None,
) -> int:
    _validate_block_rows(block_rows)
    connection = sqlite3.connect(":memory:")
    progress = _SqliteCancellation(check_cancel)
    connection.set_progress_handler(progress, 1_000)
    try:
        _create_support_table(connection)
        _count_support(connection, labels, edge_blocks, block_rows, check_cancel)
        return _apply_supported_merges(
            connection,
            labels,
            prototype_neighbors,
            min_support_pairs,
            max_members,
            block_rows,
            check_cancel,
        )
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def consolidate_labels_on_disk(
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    database_path: Path,
    min_support_pairs: int,
    max_members: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None = None,
    *,
    prototype_neighbors: np.ndarray,
) -> int:
    _validate_block_rows(block_rows)
    connection = sqlite3.connect(database_path)
    progress = _SqliteCancellation(check_cancel)
    connection.set_progress_handler(progress, 1_000)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        _create_support_table(connection)
        _count_support(connection, labels, edge_blocks, block_rows, check_cancel)
        merged_count = _apply_supported_merges(
            connection,
            labels,
            prototype_neighbors,
            min_support_pairs,
            max_members,
            block_rows,
            check_cancel,
        )
        if isinstance(labels, np.memmap):
            _check_cancel(check_cancel)
            labels.flush()
        return merged_count
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _create_support_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE support_members ("
        "left_cluster INTEGER, right_cluster INTEGER, "
        "left_member INTEGER, right_member INTEGER, "
        "PRIMARY KEY (left_cluster, right_cluster, left_member, right_member))"
    )


def _count_support(
    connection: sqlite3.Connection,
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    statement = "INSERT OR IGNORE INTO support_members VALUES (?, ?, ?, ?)"
    for block in edge_blocks:
        _check_cancel(check_cancel)
        left_clusters = labels[block.left_ids]
        right_clusters = labels[block.right_ids]
        valid = (
            (left_clusters >= 0)
            & (right_clusters >= 0)
            & (left_clusters != right_clusters)
        )
        left_first = left_clusters[valid] < right_clusters[valid]
        left_members = block.left_ids[valid]
        right_members = block.right_ids[valid]
        columns = (
            np.where(left_first, left_clusters[valid], right_clusters[valid]),
            np.where(left_first, right_clusters[valid], left_clusters[valid]),
            np.where(left_first, left_members, right_members),
            np.where(left_first, right_members, left_members),
        )
        for start in range(0, len(columns[0]), block_rows):
            _check_cancel(check_cancel)
            stop = min(start + block_rows, len(columns[0]))
            rows = zip(
                *(column[start:stop].tolist() for column in columns), strict=True
            )
            connection.executemany(statement, rows)
        connection.commit()


def _apply_supported_merges(
    connection: sqlite3.Connection,
    labels: np.ndarray,
    prototype_neighbors: np.ndarray,
    min_support_pairs: int,
    max_members: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> int:
    if min_support_pairs <= 1 or max_members <= 0:
        raise ValueError(
            "consolidation requires min_support_pairs > 1 and max_members > 0"
        )
    if len(prototype_neighbors) != len(labels):
        raise ValueError("prototype neighbors and labels must have equal length")
    sizes = _cluster_sizes(labels, block_rows, check_cancel)
    merge_map = np.arange(len(labels), dtype=np.int64)
    used = np.zeros(len(labels), dtype=np.bool_)
    rows = connection.execute(
        "SELECT left_cluster, right_cluster, COUNT(DISTINCT left_member), "
        "COUNT(DISTINCT right_member) FROM support_members "
        "GROUP BY left_cluster, right_cluster "
        "HAVING COUNT(DISTINCT left_member) >= ? "
        "AND COUNT(DISTINCT right_member) >= ? "
        "ORDER BY MIN(left_member), left_cluster, right_cluster",
        (min_support_pairs, min_support_pairs),
    )
    merged_count = 0
    for left, right, _left_count, _right_count in rows:
        _check_cancel(check_cancel)
        reciprocal = (
            prototype_neighbors[left] == right and prototype_neighbors[right] == left
        )
        if (
            reciprocal
            and not used[left]
            and not used[right]
            and (sizes[left] + sizes[right] <= max_members)
        ):
            merge_map[right] = left
            used[left] = True
            used[right] = True
            merged_count += 1
    for start in range(0, len(labels), block_rows):
        _check_cancel(check_cancel)
        stop = min(start + block_rows, len(labels))
        block = labels[start:stop]
        valid = block >= 0
        block[valid] = merge_map[block[valid]]
    return merged_count


def _cluster_sizes(
    labels: np.ndarray,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> np.ndarray:
    sizes = np.zeros(len(labels), dtype=np.int64)
    for start in range(0, len(labels), block_rows):
        _check_cancel(check_cancel)
        block = labels[start : start + block_rows]
        valid = block >= 0
        np.add.at(sizes, block[valid], 1)
    return sizes


def _check_cancel(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()


def _validate_block_rows(block_rows: int) -> None:
    if block_rows <= 0:
        raise ValueError("consolidation block_rows must be positive")


@dataclass(frozen=True)
class _SqliteCancellation:
    check_cancel: Callable[[], None] | None

    def __call__(self) -> int:
        _check_cancel(self.check_cancel)
        return 0
