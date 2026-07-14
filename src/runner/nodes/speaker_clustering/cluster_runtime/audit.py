from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from runner.nodes.models import SpeakerClusterRunRef
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import ClusteringArtifactRole


def audit_cluster_assignments(
    cluster_run: SpeakerClusterRunRef,
    database_path: Path,
    batch_rows: int,
) -> dict[str, object]:
    if database_path.exists():
        database_path.unlink()
    paths = _assignment_paths(cluster_run)
    connection = sqlite3.connect(database_path)
    outcomes: Counter[str] = Counter()
    total_rows = 0
    truth_labeled_rows = 0
    clustered_truth_rows = 0
    try:
        _create_tables(connection)
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=batch_rows,
                columns=["outcome", "cluster_id", "true_label"],
            ):
                batch_outcomes = batch.column("outcome").to_pylist()
                cluster_ids = batch.column("cluster_id").to_pylist()
                true_labels = batch.column("true_label").to_pylist()
                outcomes.update(str(outcome) for outcome in batch_outcomes)
                total_rows += batch.num_rows
                truth = [str(label) for label in true_labels if label is not None]
                clustered = [
                    (int(cluster_id), str(label))
                    for cluster_id, label in zip(cluster_ids, true_labels, strict=True)
                    if cluster_id is not None and label is not None
                ]
                truth_labeled_rows += len(truth)
                clustered_truth_rows += len(clustered)
                _record_truth(connection, truth)
                _record_clustered(connection, clustered)
                connection.commit()
        return _metrics(
            connection,
            outcomes,
            total_rows,
            truth_labeled_rows,
            clustered_truth_rows,
        )
    finally:
        connection.close()


def _assignment_paths(cluster_run: SpeakerClusterRunRef) -> list[Path]:
    with database_session() as session:
        durable = speaker_crud.list_clustering_artifacts(
            session,
            cluster_run.run_id,
            ClusteringArtifactRole.ASSIGNMENT,
        )
        artifact_ids = [artifact.artifact_id for artifact in durable]
        if artifact_ids != cluster_run.assignment_artifact_ids:
            raise ValueError(
                "cluster run assignment manifest does not match durable artifacts"
            )
        return [
            asset_crud.get_extra_file_path(session, artifact_id)
            for artifact_id in artifact_ids
        ]


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE cluster_label (cluster_id INTEGER, label TEXT, count INTEGER, "
        "PRIMARY KEY (cluster_id, label))"
    )
    connection.execute(
        "CREATE TABLE label_total (label TEXT PRIMARY KEY, count INTEGER)"
    )


def _record_clustered(
    connection: sqlite3.Connection,
    rows: list[tuple[int, str]],
) -> None:
    connection.executemany(
        "INSERT INTO cluster_label VALUES (?, ?, 1) ON CONFLICT(cluster_id, label) "
        "DO UPDATE SET count = count + 1",
        rows,
    )


def _record_truth(connection: sqlite3.Connection, labels: list[str]) -> None:
    connection.executemany(
        "INSERT INTO label_total VALUES (?, 1) ON CONFLICT(label) "
        "DO UPDATE SET count = count + 1",
        ((label,) for label in labels),
    )


def _metrics(
    connection: sqlite3.Connection,
    outcomes: Counter[str],
    total_rows: int,
    truth_labeled_rows: int,
    clustered_truth_rows: int,
) -> dict[str, object]:
    predicted_pairs = _scalar(
        connection,
        "SELECT COALESCE(SUM(total * (total - 1) / 2), 0) FROM "
        "(SELECT SUM(count) AS total FROM cluster_label GROUP BY cluster_id)",
    )
    correct_pairs = _scalar(
        connection,
        "SELECT COALESCE(SUM(count * (count - 1) / 2), 0) FROM cluster_label",
    )
    true_pairs = _scalar(
        connection,
        "SELECT COALESCE(SUM(count * (count - 1) / 2), 0) FROM label_total",
    )
    impure_clusters = _scalar(
        connection,
        "SELECT COUNT(*) FROM (SELECT cluster_id FROM cluster_label "
        "GROUP BY cluster_id HAVING COUNT(*) > 1)",
    )
    false_merge_pairs = predicted_pairs - correct_pairs
    return {
        "total_rows": total_rows,
        "truth_labeled_rows": truth_labeled_rows,
        "truth_missing_rows": total_rows - truth_labeled_rows,
        "clustered_truth_rows": clustered_truth_rows,
        "unclustered_truth_rows": truth_labeled_rows - clustered_truth_rows,
        "clustered_truth_coverage": 1.0
        if truth_labeled_rows == 0
        else clustered_truth_rows / truth_labeled_rows,
        "outcomes": dict(sorted(outcomes.items())),
        "predicted_same_speaker_pairs": predicted_pairs,
        "correct_same_speaker_pairs": correct_pairs,
        "random_speaker_collision_pairs": false_merge_pairs,
        "impure_clusters": impure_clusters,
        "pair_precision": 1.0
        if predicted_pairs == 0
        else correct_pairs / predicted_pairs,
        "pair_recall": 1.0 if true_pairs == 0 else correct_pairs / true_pairs,
    }


def _scalar(connection: sqlite3.Connection, statement: str) -> int:
    value = connection.execute(statement).fetchone()
    assert value is not None, "speaker audit aggregate returned no row"
    return int(value[0])
