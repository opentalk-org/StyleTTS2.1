from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from shared.db.statistics.clickhouse import (
    delete_statistics_entry,
    get_statistics_entry,
    list_statistics_entries,
)
from shared.db.statistics.schemas import StatisticsEntryRead, StatisticsEntrySummary

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("", response_model=list[StatisticsEntrySummary])
async def list_statistics(
    dataset_id: UUID | None = None,
) -> list[StatisticsEntrySummary]:
    return [
        StatisticsEntrySummary(
            id=item.id,
            name=item.name,
            dataset_id=item.dataset_id,
            file_count=int(item.payload["file_count"]),
            created_at=item.created_at,
        )
        for item in list_statistics_entries(dataset_id)
    ]


@router.get("/{statistics_entry_id}", response_model=StatisticsEntryRead)
async def get_statistics(statistics_entry_id: UUID) -> StatisticsEntryRead:
    try:
        return StatisticsEntryRead.model_validate(
            get_statistics_entry(statistics_entry_id)
        )
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error


@router.delete("/{statistics_entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_statistics(statistics_entry_id: UUID) -> None:
    try:
        get_statistics_entry(statistics_entry_id)
        delete_statistics_entry(statistics_entry_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
