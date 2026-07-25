from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path
import sqlite3

import numpy as np

from runner.nodes.speaker_clustering.edge_shards import EdgeBlock


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
    _validate_inputs(labels, prototype_neighbors, block_rows)
    connection = sqlite3.connect(database_path)
    progress = partial(_sqlite_progress, check_cancel)
    connection.set_progress_handler(progress, 1_000)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        _create_support_tables(connection)
        _count_support(
            connection,
            labels,
            edge_blocks,
            prototype_neighbors,
            min_support_pairs,
            block_rows,
            check_cancel,
        )
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


def _create_support_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE left_support ("
        "left_cluster INTEGER, right_cluster INTEGER, "
        "member INTEGER, PRIMARY KEY (left_cluster, right_cluster, member))"
    )
    connection.execute(
        "CREATE TABLE right_support ("
        "left_cluster INTEGER, right_cluster INTEGER, "
        "member INTEGER, PRIMARY KEY (left_cluster, right_cluster, member))"
    )


def _count_support(
    connection: sqlite3.Connection,
    labels: np.ndarray,
    edge_blocks: Iterable[EdgeBlock],
    prototype_neighbors: np.ndarray,
    min_support_pairs: int,
    block_rows: int,
    check_cancel: Callable[[], None] | None,
) -> None:
    statement = (
        "INSERT OR IGNORE INTO {table} SELECT ?, ?, ? WHERE "
        "(SELECT COUNT(*) FROM {table} WHERE left_cluster = ? "
        "AND right_cluster = ?) < ?"
    )
    for block in edge_blocks:
        _check_cancel(check_cancel)
        left_clusters = labels[block.left_ids]
        right_clusters = labels[block.right_ids]
        valid = (
            (left_clusters >= 0)
            & (right_clusters >= 0)
            & (left_clusters != right_clusters)
        )
        safe_left = np.where(valid, left_clusters, 0)
        safe_right = np.where(valid, right_clusters, 0)
        valid &= (prototype_neighbors[safe_left] == safe_right) & (
            prototype_neighbors[safe_right] == safe_left
        )
        left_first = left_clusters[valid] < right_clusters[valid]
        left_members = block.left_ids[valid]
        right_members = block.right_ids[valid]
        cluster_left = np.where(left_first, left_clusters[valid], right_clusters[valid])
        cluster_right = np.where(
            left_first, right_clusters[valid], left_clusters[valid]
        )
        member_left = np.where(left_first, left_members, right_members)
        member_right = np.where(left_first, right_members, left_members)
        for start in range(0, len(cluster_left), block_rows):
            _check_cancel(check_cancel)
            stop = min(start + block_rows, len(cluster_left))
            pair_columns = (cluster_left[start:stop], cluster_right[start:stop])
            left_rows = _support_rows(
                pair_columns, member_left[start:stop], min_support_pairs
            )
            right_rows = _support_rows(
                pair_columns, member_right[start:stop], min_support_pairs
            )
            _executemany(
                connection,
                statement.format(table="left_support"),
                left_rows,
                check_cancel,
            )
            _executemany(
                connection,
                statement.format(table="right_support"),
                right_rows,
                check_cancel,
            )
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
    sizes = _cluster_sizes(labels, block_rows, check_cancel)
    merge_map = np.arange(len(labels), dtype=np.int64)
    used = np.zeros(len(labels), dtype=np.bool_)
    rows = _sqlite_rows(
        connection,
        "SELECT left_members.left_cluster, left_members.right_cluster "
        "FROM (SELECT left_cluster, right_cluster, MIN(member) AS first_member "
        "FROM left_support GROUP BY left_cluster, right_cluster HAVING COUNT(*) >= ?) "
        "AS left_members JOIN (SELECT left_cluster, right_cluster FROM right_support "
        "GROUP BY left_cluster, right_cluster HAVING COUNT(*) >= ?) AS right_members "
        "USING (left_cluster, right_cluster) ORDER BY left_members.first_member, "
        "left_members.left_cluster, left_members.right_cluster",
        (min_support_pairs, min_support_pairs),
        check_cancel,
    )
    merged_count = 0
    for left, right in rows:
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
        block = labels[start : start + block_rows]
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


def _validate_inputs(
    labels: np.ndarray, prototype_neighbors: np.ndarray, block_rows: int
) -> None:
    if block_rows <= 0:
        raise ValueError("consolidation block_rows must be positive")
    if len(prototype_neighbors) != len(labels):
        raise ValueError("prototype neighbors and labels must have equal length")


def _support_rows(
    pair_columns: tuple[np.ndarray, np.ndarray],
    members: np.ndarray,
    min_support_pairs: int,
) -> zip:
    left, right = pair_columns
    columns = (left.tolist(), right.tolist(), members.tolist())
    counts = [min_support_pairs] * len(left)
    return zip(*columns, columns[0], columns[1], counts, strict=True)


def _executemany(
    connection: sqlite3.Connection, statement: str,
    rows: Iterable[tuple[int, int, int, int, int, int]],
    check_cancel: Callable[[], None] | None,
) -> None:
    try:
        connection.executemany(statement, rows)
    except sqlite3.OperationalError as error:
        _translate_sqlite_interrupt(error, check_cancel)


def _sqlite_rows(
    connection: sqlite3.Connection, statement: str, parameters: tuple[int, int],
    check_cancel: Callable[[], None] | None,
) -> Iterator[tuple[int, int]]:
    try:
        yield from connection.execute(statement, parameters)
    except sqlite3.OperationalError as error:
        _translate_sqlite_interrupt(error, check_cancel)


def _translate_sqlite_interrupt(
    error: sqlite3.OperationalError, check_cancel: Callable[[], None] | None
) -> None:
    if error.args == ("interrupted",):
        _check_cancel(check_cancel)
    raise error


def _sqlite_progress(check_cancel: Callable[[], None] | None) -> int:
    _check_cancel(check_cancel)
    return 0
