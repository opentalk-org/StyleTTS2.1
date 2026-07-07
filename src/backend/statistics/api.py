from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from shared.db import database_session
from shared.db.statistics import crud
from shared.db.statistics.schemas import StatisticsEntryRead, StatisticsEntrySummary

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("", response_model=list[StatisticsEntrySummary])
async def list_statistics(dataset_id: UUID | None = None) -> list[StatisticsEntrySummary]:
    with database_session() as session:
        return [StatisticsEntrySummary.model_validate(item) for item in crud.list_statistics_summaries(session, dataset_id)]


@router.get("/{statistics_entry_id}", response_model=StatisticsEntryRead)
async def get_statistics(statistics_entry_id: UUID) -> StatisticsEntryRead:
    try:
        with database_session() as session:
            return StatisticsEntryRead.model_validate(crud.get_statistics_entry(session, statistics_entry_id))
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.delete("/{statistics_entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_statistics(statistics_entry_id: UUID) -> None:
    try:
        with database_session() as session:
            crud.delete_statistics_entry(session, statistics_entry_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
