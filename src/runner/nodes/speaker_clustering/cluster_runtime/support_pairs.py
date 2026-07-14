from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import sqlite3

import numpy as np

from runner.nodes.speaker_clustering.edge_shards import EdgeBlock


def consolidate_labels(
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    min_support_pairs: int,
    max_members: int,
) -> np.ndarray:
    connection = sqlite3.connect(":memory:")
    try:
        _create_support_table(connection)
        _count_support(connection, labels, edge_blocks, None)
        _apply_supported_merges(connection, labels, min_support_pairs, max_members)
        return labels
    finally:
        connection.close()


def consolidate_labels_on_disk(
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    database_path: Path,
    min_support_pairs: int,
    max_members: int,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        _create_support_table(connection)
        _count_support(connection, labels, edge_blocks, check_cancel)
        _apply_supported_merges(connection, labels, min_support_pairs, max_members)
        if isinstance(labels, np.memmap):
            labels.flush()
    finally:
        connection.close()


def _create_support_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE support (left_cluster INTEGER, right_cluster INTEGER, "
        "pair_count INTEGER, PRIMARY KEY (left_cluster, right_cluster))"
    )


def _count_support(
    connection: sqlite3.Connection,
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    check_cancel: Callable[[], None] | None,
) -> None:
    statement = (
        "INSERT INTO support VALUES (?, ?, 1) ON CONFLICT(left_cluster, right_cluster) "
        "DO UPDATE SET pair_count = pair_count + 1"
    )
    for block in edge_blocks:
        if check_cancel is not None:
            check_cancel()
        left = labels[block.left_ids]
        right = labels[block.right_ids]
        valid = (left >= 0) & (right >= 0) & (left != right)
        low = np.minimum(left[valid], right[valid])
        high = np.maximum(left[valid], right[valid])
        connection.executemany(statement, zip(low.tolist(), high.tolist(), strict=True))
        connection.commit()


def _apply_supported_merges(
    connection: sqlite3.Connection,
    labels: np.ndarray,
    min_support_pairs: int,
    max_members: int,
) -> None:
    if min_support_pairs <= 1 or max_members <= 0:
        raise ValueError(
            "consolidation requires min_support_pairs > 1 and max_members > 0"
        )
    valid = labels >= 0
    sizes = np.bincount(labels[valid], minlength=len(labels))
    merge_map = np.arange(len(labels), dtype=np.int64)
    used = np.zeros(len(labels), dtype=np.bool_)
    rows = connection.execute(
        "SELECT left_cluster, right_cluster, pair_count FROM support "
        "WHERE pair_count >= ? ORDER BY pair_count DESC, left_cluster, right_cluster",
        (min_support_pairs,),
    )
    for left, right, _count in rows:
        if used[left] or used[right] or sizes[left] + sizes[right] > max_members:
            continue
        merge_map[right] = left
        used[left] = True
        used[right] = True
    labels[valid] = merge_map[labels[valid]]
