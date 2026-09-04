from collections.abc import Sequence
from datetime import datetime, timedelta
from uuid import UUID

from shared.db.audio.clickhouse.files import get_audio_files
from shared.db.clickhouse import clickhouse_client, delete_rows
from shared.db.clickhouse.types import utc_datetime
from shared.db.datasets.clickhouse.models import DatasetMembership, DatasetRecord


def create_dataset(item: DatasetRecord) -> DatasetRecord:
    existing = clickhouse_client().query(
        """
        SELECT id
        FROM datasets FINAL
        WHERE name = {name:String}
        LIMIT 1
        """,
        parameters={"name": item.name},
    )
    if existing.result_rows:
        raise ValueError(f"Dataset name already exists: {item.name}")
    clickhouse_client().insert(
        "datasets",
        [[item.id, item.updated_at, item.name]],
        column_names=["id", "updated_at", "name"],
    )
    return get_dataset(item.id)


def get_dataset(dataset_id: UUID) -> DatasetRecord:
    result = clickhouse_client().query(
        """
        SELECT id, updated_at, name
        FROM datasets FINAL
        WHERE id = {id:UUID}
        """,
        parameters={"id": dataset_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Dataset not found: {dataset_id}")
    return DatasetRecord.model_validate(rows[0])


def list_datasets() -> list[DatasetRecord]:
    result = clickhouse_client().query(
        """
        SELECT id, updated_at, name
        FROM datasets FINAL
        ORDER BY name, id
        """
    )
    return [DatasetRecord.model_validate(row) for row in result.named_results()]


def list_dataset_file_counts() -> list[tuple[DatasetRecord, int]]:
    counts = clickhouse_client().query(
        """
        SELECT dataset_id, count() AS file_count
        FROM dataset_audio_files FINAL
        GROUP BY dataset_id
        """
    )
    by_dataset = {row[0]: int(row[1]) for row in counts.result_rows}
    return [(dataset, by_dataset.get(dataset.id, 0)) for dataset in list_datasets()]


def dataset_ids_by_audio_file(
    audio_file_ids: Sequence[UUID],
) -> dict[UUID, list[UUID]]:
    ids = list(dict.fromkeys(audio_file_ids))
    result = clickhouse_client().query(
        """
        SELECT audio_file_id, groupArray(dataset_id) AS dataset_ids
        FROM dataset_audio_files FINAL
        WHERE audio_file_id IN {ids:Array(UUID)}
        GROUP BY audio_file_id
        """,
        parameters={"ids": ids},
    )
    memberships = {row[0]: list(row[1]) for row in result.result_rows}
    return {audio_id: memberships.get(audio_id, []) for audio_id in ids}


def add_audio_files(
    dataset_id: UUID,
    audio_file_ids: Sequence[UUID],
    updated_at: datetime,
    created_at: datetime,
) -> None:
    get_dataset(dataset_id)
    if not audio_file_ids:
        return
    ids = list(dict.fromkeys(audio_file_ids))
    existing_ids = {item.id for item in get_audio_files(ids)}
    missing = set(ids).difference(existing_ids)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(map(str, missing))}")
    current_result = clickhouse_client().query(
        """
        SELECT audio_file_id, updated_at, created_at
        FROM dataset_audio_files FINAL
        WHERE dataset_id = {dataset_id:UUID}
          AND audio_file_id IN {ids:Array(UUID)}
        """,
        parameters={"dataset_id": dataset_id, "ids": ids},
    )
    current = {
        row[0]: (utc_datetime(row[1]), utc_datetime(row[2]))
        for row in current_result.result_rows
    }
    rows = [
        DatasetMembership(
            dataset_id=dataset_id,
            audio_file_id=audio_id,
            updated_at=max(
                updated_at,
                current[audio_id][0] + timedelta(microseconds=1),
            )
            if audio_id in current
            else updated_at,
            created_at=current[audio_id][1]
            if audio_id in current
            else created_at,
        )
        for audio_id in ids
    ]
    clickhouse_client().insert(
        "dataset_audio_files",
        [
            [row.dataset_id, row.audio_file_id, row.updated_at, row.created_at]
            for row in rows
        ],
        column_names=["dataset_id", "audio_file_id", "updated_at", "created_at"],
    )


def remove_audio_files(dataset_id: UUID, audio_file_ids: Sequence[UUID]) -> None:
    if not audio_file_ids:
        return
    delete_rows(
        clickhouse_client(),
        "dataset_audio_files",
        "dataset_id = {dataset_id:UUID} AND audio_file_id IN {ids:Array(UUID)}",
        {"dataset_id": dataset_id, "ids": list(audio_file_ids)},
    )


def delete_dataset(dataset_id: UUID) -> None:
    get_dataset(dataset_id)
    delete_rows(
        clickhouse_client(),
        "statistics_entries",
        "dataset_id = {id:UUID}",
        {"id": dataset_id},
    )
    delete_rows(
        clickhouse_client(),
        "dataset_audio_files",
        "dataset_id = {id:UUID}",
        {"id": dataset_id},
    )
    delete_rows(
        clickhouse_client(),
        "datasets",
        "id = {id:UUID}",
        {"id": dataset_id},
    )
