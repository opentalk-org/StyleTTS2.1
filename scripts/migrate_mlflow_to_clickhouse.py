import argparse
import json
import multiprocessing
import os
import traceback
from collections.abc import Callable, Iterator, Sequence
from ctypes import c_float
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import clickhouse_connect
from sqlalchemy import create_engine, text
from tqdm import tqdm

from migrate_mlflow_artifacts import (
    backfill_inline_artifacts,
    migrate_artifacts,
    reconcile_checkpoint_runs,
)


MLFLOW_DATABASE_URL_ENV = "MLFLOW_DATABASE_URL"
CLICKHOUSE_URL_ENV = "RUNFLOW_CLICKHOUSE_URL"
DEFAULT_BATCH_SIZE = 50_000
Row = Sequence[Any]
Transform = Callable[[Row], Row]


@dataclass(frozen=True)
class CopySpec:
    destination: str
    columns: tuple[str, ...]
    source_sql: str
    source_count_sql: str
    partition_key: str
    transform: Transform


def project_id(experiment_id: object) -> UUID:
    return uuid5(NAMESPACE_URL, f"mlflow-experiment:{experiment_id}")


def project_row(row: Row) -> Row:
    values = list(row)
    values[0] = project_id(values[0])
    return values


def run_row(row: Row) -> Row:
    values = list(row)
    values[0] = UUID(values[0])
    values[1] = project_id(values[1])
    values[3] = json.dumps(values[3], separators=(",", ":"))
    values[4] = json.dumps(values[4], separators=(",", ":"))
    return values


def referenced_row(row: Row) -> Row:
    values = list(row)
    values[1] = UUID(values[1])
    return values


def metric_row(row: Row) -> Row:
    values = list(referenced_row(row))
    values[4] = c_float(values[4]).value
    return values


def specs() -> tuple[CopySpec, ...]:
    return (
        CopySpec(
            destination="projects",
            columns=("id", "name", "description", "created_at", "updated_at"),
            source_sql="""
            SELECT experiment_id, name, artifact_location,
                   to_timestamp(creation_time / 1000.0),
                   to_timestamp(last_update_time / 1000.0)
            FROM experiments
            WHERE lifecycle_stage = 'active'
            """,
            source_count_sql="""
            SELECT count(*) FROM experiments WHERE lifecycle_stage = 'active'
            """,
            partition_key="experiment_id",
            transform=project_row,
        ),
        CopySpec(
            destination="runs",
            columns=("id", "project_id", "name", "data_config", "train_config"),
            source_sql="""
            SELECT r.run_uuid, r.experiment_id, r.name,
                   jsonb_build_object(
                       'artifact_uri', r.artifact_uri,
                       'tags', COALESCE(t.values, '{}'::jsonb)
                   ),
                   COALESCE(p.values, '{}'::jsonb)
            FROM runs AS r
            LEFT JOIN (
                SELECT run_uuid, jsonb_object_agg(key, value) AS values
                FROM tags GROUP BY run_uuid
            ) AS t ON t.run_uuid = r.run_uuid
            LEFT JOIN (
                SELECT run_uuid, jsonb_object_agg(key, value) AS values
                FROM params GROUP BY run_uuid
            ) AS p ON p.run_uuid = r.run_uuid
            WHERE r.lifecycle_stage = 'active'
            """,
            source_count_sql="""
            SELECT count(*) FROM runs WHERE lifecycle_stage = 'active'
            """,
            partition_key="run_uuid",
            transform=run_row,
        ),
        CopySpec(
            destination="run_status",
            columns=("timestamp", "run_id", "status"),
            source_sql="""
            SELECT to_timestamp(start_time / 1000.0), run_uuid, 'running'
            FROM runs
            WHERE lifecycle_stage = 'active'
            UNION ALL
            SELECT to_timestamp(end_time / 1000.0), run_uuid,
                   CASE status
                       WHEN 'FINISHED' THEN 'succeeded'
                       WHEN 'FAILED' THEN 'failed'
                       WHEN 'KILLED' THEN 'cancelled'
                   END
            FROM runs
            WHERE lifecycle_stage = 'active'
              AND end_time IS NOT NULL
              AND status != 'RUNNING'
            """,
            source_count_sql="""
            SELECT count(*) + count(*) FILTER (
                WHERE end_time IS NOT NULL AND status != 'RUNNING'
            )
            FROM runs WHERE lifecycle_stage = 'active'
            """,
            partition_key="run_uuid",
            transform=referenced_row,
        ),
        CopySpec(
            destination="metrics",
            columns=("timestamp", "run_id", "step", "name", "value"),
            source_sql="""
            SELECT to_timestamp(m.timestamp / 1000.0), m.run_uuid,
                   m.step, m.key, m.value
            FROM metrics AS m
            JOIN runs AS r ON r.run_uuid = m.run_uuid
            WHERE r.lifecycle_stage = 'active'
            """,
            source_count_sql="""
            SELECT count(*)
            FROM metrics AS m
            JOIN runs AS r ON r.run_uuid = m.run_uuid
            WHERE r.lifecycle_stage = 'active'
            """,
            partition_key="run_uuid",
            transform=metric_row,
        ),
    )


def batches(result: Any, size: int, transform: Transform) -> Iterator[list[Row]]:
    while rows := result.fetchmany(size):
        yield [transform(row) for row in rows]


def copy_partition(
    spec: CopySpec, worker: int, workers: int, batch_size: int, updates: Any
) -> None:
    source_engine = create_engine(os.environ[MLFLOW_DATABASE_URL_ENV])
    destination = clickhouse_connect.get_client(dsn=os.environ[CLICKHOUSE_URL_ENV])
    query = f"""
        SELECT * FROM ({spec.source_sql}) AS migration_source
        WHERE (hashtextextended({spec.partition_key}::text, 0)
               & 9223372036854775807) % :workers = :worker
    """
    copied = 0
    try:
        source = source_engine.connect()
        source.execute(text("SET TRANSACTION READ ONLY"))
        result = source.execution_options(stream_results=True).execute(
            text(query), {"workers": workers, "worker": worker}
        )
        for batch in batches(result, batch_size, spec.transform):
            destination.insert(spec.destination, batch, column_names=spec.columns)
            copied += len(batch)
            updates.put(("progress", len(batch)))
        source.close()
        updates.put(("done", copied))
    except BaseException:
        updates.put(("error", traceback.format_exc()))


def copy_table(spec: CopySpec, batch_size: int, workers: int) -> int:
    source_engine = create_engine(os.environ[MLFLOW_DATABASE_URL_ENV])
    with source_engine.connect() as source:
        source.execute(text("SET TRANSACTION READ ONLY"))
        expected = source.execute(text(spec.source_count_sql)).scalar_one()
    context = multiprocessing.get_context("spawn")
    updates = context.Queue()
    processes = [
        context.Process(
            target=copy_partition,
            args=(spec, worker, workers, batch_size, updates),
        )
        for worker in range(workers)
    ]
    for process in processes:
        process.start()
    copied = completed = 0
    with tqdm(total=expected, desc=spec.destination, unit="rows") as progress:
        while completed < workers:
            kind, value = updates.get()
            match kind:
                case "progress":
                    copied += value
                    progress.update(value)
                case "done":
                    completed += 1
                case "error":
                    for process in processes:
                        process.terminate()
                    raise RuntimeError(value)
    for process in processes:
        process.join()
    assert copied == expected, f"source count changed for {spec.destination}"
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inline-artifacts-only", action="store_true")
    args = parser.parse_args()
    migration_specs = specs()
    destination = clickhouse_connect.get_client(dsn=os.environ[CLICKHOUSE_URL_ENV])
    if args.inline_artifacts_only:
        backfill_inline_artifacts(destination)
        return
    for spec in reversed(migration_specs):
        destination.command(f"TRUNCATE TABLE {spec.destination}")
        print(f"reset {spec.destination}", flush=True)
    expected = {
        spec.destination: copy_table(spec, args.batch_size, args.workers)
        for spec in migration_specs
    }
    for table, source_count in expected.items():
        destination_count = destination.query(f"SELECT count() FROM {table}").first_row[
            0
        ]
        assert destination_count == source_count
        print(f"validated {table}: {destination_count}", flush=True)
    migrate_artifacts(destination, args.batch_size)
    reconcile_checkpoint_runs(destination)


if __name__ == "__main__":
    main()
