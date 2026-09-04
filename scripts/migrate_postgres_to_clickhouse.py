import argparse
import json
import multiprocessing
import os
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from sqlalchemy import create_engine, text
from tqdm import tqdm


PG_URL_ENV = "RUNFLOW_PGBOUNCER_DATABASE_URL"
CLICKHOUSE_URL_ENV = "RUNFLOW_CLICKHOUSE_URL"
DEFAULT_BATCH_SIZE = 50_000
EMPTY_UUID = UUID(int=0)
Row = Sequence[Any]
Transform = Callable[[Row], Row]


@dataclass(frozen=True)
class CopySpec:
    destination: str
    columns: tuple[str, ...]
    source_sql: str
    source_count_sql: str
    partition_key: str
    transform: Transform = tuple


def serialize_json_fields(row: Row, indexes: tuple[int, ...]) -> Row:
    values = list(row)
    for index in indexes:
        values[index] = json.dumps(values[index], separators=(",", ":"))
    return values


def asset_row(row: Row) -> Row:
    values = list(serialize_json_fields(row, indexes=(8,)))
    values[9] = values[9] or EMPTY_UUID
    values[10] = values[10] or EMPTY_UUID
    return values


def config_row(row: Row) -> Row:
    return serialize_json_fields(row, indexes=(4,))


def audio_file_row(row: Row) -> Row:
    values = list(serialize_json_fields(row, indexes=(13, 14)))
    values[3] = values[3] or EMPTY_UUID
    values[7] = values[7] if values[7] is not None else float("inf")
    for index in (8, 9, 10):
        values[index] = values[index] or ""
    return values


def audio_segment_row(row: Row) -> Row:
    values = list(serialize_json_fields(row, indexes=(11,)))
    values[9] = values[9] if values[9] is not None else -1.0
    values[10] = values[10] or ""
    values[12] = [
        (item["word"], float(item["start"]), float(item["end"]))
        for item in (values[12] or [])
    ]
    return values


def statistics_row(row: Row) -> Row:
    values = list(serialize_json_fields(row, indexes=(4, 5)))
    values[3] = values[3] or EMPTY_UUID
    return values


def specs(migrated_at: datetime) -> tuple[CopySpec, ...]:
    timestamp = migrated_at.isoformat()
    return (
        CopySpec(
            destination="bucket_files",
            columns=("id", "kind", "path", "size", "used_bytes"),
            source_sql="""
            SELECT id, 'audio', path, size, used_bytes FROM bucket_files
            UNION ALL
            SELECT id, 'waveform', path, size, used_bytes FROM waveform_packs
            """,
            source_count_sql="""
            SELECT
                (SELECT count(*) FROM bucket_files)
                + (SELECT count(*) FROM waveform_packs)
            """,
            partition_key="id",
        ),
        CopySpec(
            destination="assets",
            columns=(
                "id",
                "updated_at",
                "kind",
                "name",
                "path",
                "size",
                "content_hash",
                "type",
                "metadata",
                "run_id",
                "ancestor_asset_id",
            ),
            source_sql="""
            SELECT checkpoint.id, TIMESTAMPTZ '__TIMESTAMP__', 'checkpoint',
                   checkpoint.name, checkpoint.path, checkpoint.size,
                   checkpoint.content_hash, checkpoint.type, checkpoint.metadata,
                   CASE WHEN checkpoint.job_id ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
                        THEN checkpoint.job_id::uuid ELSE NULL END,
                   COALESCE(CASE
                       WHEN checkpoint.metadata->>'ancestor_asset_id' ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
                       THEN (checkpoint.metadata->>'ancestor_asset_id')::uuid
                       WHEN checkpoint.metadata->>'base_checkpoint_id' ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
                       THEN (checkpoint.metadata->>'base_checkpoint_id')::uuid
                       ELSE NULL
                   END, selected_checkpoint.id)
            FROM checkpoints AS checkpoint
            LEFT JOIN jobs AS job ON job.run_id = checkpoint.job_id
            LEFT JOIN LATERAL (
                SELECT (array_agg(DISTINCT value #>> '{}'))[1]::uuid AS id
                FROM jsonb_path_query(
                    job.graph_request,
                    '$.data.nodes[*] ? (@.type == "SelectCheckpoint").params.checkpoint_id'
                ) AS selected(value)
                WHERE value #>> '{}' ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
                  AND value #>> '{}' != '00000000-0000-0000-0000-000000000000'
                HAVING count(DISTINCT value #>> '{}') = 1
            ) AS selected_checkpoint ON true
            UNION ALL
            SELECT id, TIMESTAMPTZ '__TIMESTAMP__', 'file', name, path, size,
                   content_hash, type, metadata, NULL, NULL
            FROM extra_files
            """.replace("__TIMESTAMP__", timestamp),
            source_count_sql="""
            SELECT
                (SELECT count(*) FROM checkpoints)
                + (SELECT count(*) FROM extra_files)
            """,
            partition_key="id",
            transform=asset_row,
        ),
        CopySpec(
            destination="configs",
            columns=("id", "updated_at", "name", "type", "metadata"),
            source_sql=f"""
            SELECT id, TIMESTAMPTZ '{timestamp}', name, type, metadata
            FROM configs
            """,
            source_count_sql="SELECT count(*) FROM configs",
            partition_key="id",
            transform=config_row,
        ),
        CopySpec(
            destination="audio_files",
            columns=(
                "id",
                "updated_at",
                "name",
                "bucket_file_id",
                "byte_offset",
                "duration",
                "byte_length",
                "score",
                "language",
                "style_prompt",
                "voice_prompt",
                "virtual",
                "storage_kind",
                "storage_ref",
                "metadata",
            ),
            source_sql="""
            SELECT id, updated_at, name, bucket_file_id, byte_offset, duration,
                   byte_length, score, language, style_prompt, voice_prompt, virtual,
                   storage_kind, storage_ref, metadata
            FROM audio_files
            """,
            source_count_sql="SELECT count(*) FROM audio_files",
            partition_key="id",
            transform=audio_file_row,
        ),
        CopySpec(
            destination="audio_segments",
            columns=(
                "id",
                "audio_file_id",
                "updated_at",
                "position",
                "start_seconds",
                "end_seconds",
                "text",
                "phon",
                "kind",
                "accuracy",
                "speaker_id",
                "metadata",
                "alignment",
            ),
            source_sql="""
            SELECT s.id, s.audio_file_id, f.updated_at, s.position,
                   s.start_seconds, s.end_seconds, s.text, s.phon, s.kind, s.accuracy,
                   s.speaker_id, s.metadata, a.data
            FROM segments AS s
            JOIN audio_files AS f ON f.id = s.audio_file_id
            JOIN alignments AS a ON a.segment_id = s.id
            """,
            source_count_sql="SELECT count(*) FROM segments",
            partition_key="audio_file_id",
            transform=audio_segment_row,
        ),
        CopySpec(
            destination="datasets",
            columns=("id", "updated_at", "name"),
            source_sql=f"""
            SELECT id, TIMESTAMPTZ '{timestamp}', name
            FROM datasets
            """,
            source_count_sql="SELECT count(*) FROM datasets",
            partition_key="id",
        ),
        CopySpec(
            destination="dataset_audio_files",
            columns=("dataset_id", "audio_file_id", "updated_at"),
            source_sql=f"""
            SELECT dataset_id, audio_file_id, TIMESTAMPTZ '{timestamp}'
            FROM dataset_audio_files
            """,
            source_count_sql="SELECT count(*) FROM dataset_audio_files",
            partition_key="audio_file_id",
        ),
        CopySpec(
            destination="audio_waveforms",
            columns=(
                "audio_file_id",
                "updated_at",
                "pack_id",
                "byte_offset",
                "byte_length",
                "duration",
                "sample_rate",
                "points_per_second",
                "point_count",
            ),
            source_sql="""
            SELECT audio_file_id, updated_at, pack_id, byte_offset, byte_length,
                   duration, sample_rate, points_per_second, point_count
            FROM audio_waveforms
            """,
            source_count_sql="SELECT count(*) FROM audio_waveforms",
            partition_key="audio_file_id",
        ),
        CopySpec(
            destination="mos_comparisons",
            columns=(
                "id",
                "updated_at",
                "audio_a_id",
                "audio_b_id",
                "preferred_audio_id",
                "score_a",
                "score_b",
                "created_at",
            ),
            source_sql="""
            SELECT id, created_at, audio_a_id, audio_b_id, preferred_audio_id,
                   score_a, score_b, created_at
            FROM mos_comparisons
            """,
            source_count_sql="SELECT count(*) FROM mos_comparisons",
            partition_key="id",
        ),
        CopySpec(
            destination="statistics_entries",
            columns=(
                "id",
                "updated_at",
                "name",
                "dataset_id",
                "payload",
                "metadata",
                "created_at",
            ),
            source_sql="""
            SELECT id, created_at, name, dataset_id, payload, metadata, created_at
            FROM statistics_entries
            """,
            source_count_sql="SELECT count(*) FROM statistics_entries",
            partition_key="id",
            transform=statistics_row,
        ),
    )


def batches(result: Any, size: int, transform: Transform) -> Iterator[list[Row]]:
    while rows := result.fetchmany(size):
        yield [transform(row) for row in rows]


def reset_destination(client: Client, tables: Sequence[str]) -> None:
    for table in reversed(tables):
        client.command(f"TRUNCATE TABLE {table}")
        assert client.query(f"SELECT count() FROM {table}").first_row[0] == 0
        print(f"reset {table}", flush=True)


def copy_partition(
    spec: CopySpec, worker: int, workers: int, batch_size: int, updates: Any
) -> None:
    source_engine = create_engine(os.environ[PG_URL_ENV])
    destination = clickhouse_connect.get_client(dsn=os.environ[CLICKHOUSE_URL_ENV])
    partitioned_sql = f"""
        SELECT * FROM ({spec.source_sql}) AS migration_source
        WHERE (hashtextextended({spec.partition_key}::text, 0)
               & 9223372036854775807) % :workers = :worker
    """
    copied = 0
    try:
        source = source_engine.connect()
        source.execute(text("SET TRANSACTION READ ONLY"))
        result = source.execution_options(stream_results=True).execute(
            text(partitioned_sql), {"workers": workers, "worker": worker}
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
    source_engine = create_engine(os.environ[PG_URL_ENV])
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


def validate(destination: Client, expected: dict[str, int]) -> None:
    for table, source_count in expected.items():
        destination_count = destination.query(f"SELECT count() FROM {table}").first_row[
            0
        ]
        assert destination_count == source_count, (
            f"{table}: PostgreSQL={source_count}, ClickHouse={destination_count}"
        )
        print(f"validated {table}: {destination_count}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migration_specs = specs(datetime.now(UTC))
    tables = [spec.destination for spec in migration_specs]
    destination = clickhouse_connect.get_client(dsn=os.environ[CLICKHOUSE_URL_ENV])
    reset_destination(destination, tables)
    expected = {
        spec.destination: copy_table(spec, args.batch_size, args.workers)
        for spec in migration_specs
    }
    validate(destination, expected)
    print("migration complete", flush=True)


if __name__ == "__main__":
    main()
