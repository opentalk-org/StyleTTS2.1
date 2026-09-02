from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from shared.db.statistics.clickhouse.crud import (
    create_statistics_entries,
    delete_statistics_entry as _delete_entry,
    get_statistics_entry as _get_entry,
    list_statistics_entries as _list_entries,
)
from shared.db.statistics.clickhouse.models import StatisticsEntryRecord
from shared.db.statistics.schemas import StatisticsEntryCreate


def create_statistics_entry(payload: StatisticsEntryCreate) -> StatisticsEntryRecord:
    return bulk_create_statistics_entries([payload])[0]


def bulk_create_statistics_entries(
    payloads: Sequence[StatisticsEntryCreate],
) -> list[StatisticsEntryRecord]:
    now = datetime.now(UTC)
    records = [
        StatisticsEntryRecord(
            id=uuid4(), updated_at=now, created_at=now, **payload.model_dump()
        )
        for payload in payloads
    ]
    create_statistics_entries(records)
    return records


def get_statistics_entry(statistics_entry_id: UUID) -> StatisticsEntryRecord:
    return _get_entry(statistics_entry_id)


def list_statistics_entries(
    dataset_id: UUID | None = None,
) -> list[StatisticsEntryRecord]:
    return _list_entries(dataset_id)


def list_statistics_summaries(
    dataset_id: UUID | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "name": item.name,
            "dataset_id": item.dataset_id,
            "created_at": item.created_at,
            "file_count": int(item.payload.get("file_count", 0)),
        }
        for item in _list_entries(dataset_id)
    ]


def delete_statistics_entry(statistics_entry_id: UUID) -> None:
    _get_entry(statistics_entry_id)
    _delete_entry(statistics_entry_id)
