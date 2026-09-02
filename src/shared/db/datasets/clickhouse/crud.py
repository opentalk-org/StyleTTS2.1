from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from shared.db.clickhouse import clickhouse_client, delete_rows
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


def add_audio_files(
    dataset_id: UUID,
    audio_file_ids: Sequence[UUID],
    updated_at: datetime,
    created_at: datetime,
) -> None:
    get_dataset(dataset_id)
    if not audio_file_ids:
        return
    rows = [
        DatasetMembership(
            dataset_id=dataset_id,
            audio_file_id=audio_id,
            updated_at=updated_at,
            created_at=created_at,
        )
        for audio_id in dict.fromkeys(audio_file_ids)
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
