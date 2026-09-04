from collections.abc import Sequence
from uuid import UUID

from shared.db.clickhouse import clickhouse_client, delete_rows
from shared.db.statistics.clickhouse.models import StatisticsEntryRecord


def create_statistics_entries(items: Sequence[StatisticsEntryRecord]) -> None:
    if not items:
        return
    rows = [
        [
            item.id,
            item.updated_at,
            item.name,
            item.dataset_id,
            item.payload,
            item.metadata,
            item.created_at,
        ]
        for item in items
    ]
    clickhouse_client().insert(
        "statistics_entries",
        rows,
        column_names=[
            "id",
            "updated_at",
            "name",
            "dataset_id",
            "payload",
            "metadata",
            "created_at",
        ],
    )


def get_statistics_entry(entry_id: UUID) -> StatisticsEntryRecord:
    result = clickhouse_client().query(
        """
        SELECT id, updated_at, name, dataset_id, payload, metadata, created_at
        FROM statistics_entries FINAL
        WHERE id = {id:UUID}
        """,
        parameters={"id": entry_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Statistics entry not found: {entry_id}")
    return StatisticsEntryRecord.model_validate(rows[0])


def list_statistics_entries(
    dataset_id: UUID | None = None,
) -> list[StatisticsEntryRecord]:
    where = "WHERE dataset_id = {dataset_id:UUID}" if dataset_id is not None else ""
    result = clickhouse_client().query(
        f"""
        SELECT id, updated_at, name, dataset_id, payload, metadata, created_at
        FROM statistics_entries FINAL
        {where}
        ORDER BY created_at DESC, id DESC
        """,
        parameters={"dataset_id": dataset_id} if dataset_id is not None else None,
    )
    return [StatisticsEntryRecord.model_validate(row) for row in result.named_results()]


def delete_statistics_entry(entry_id: UUID) -> None:
    delete_rows(
        clickhouse_client(),
        "statistics_entries",
        "id = {id:UUID}",
        {"id": entry_id},
    )
