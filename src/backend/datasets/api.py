import uuid

from fastapi import APIRouter, HTTPException, status

from shared.db import database_session
from shared.db.datasets import crud as dataset_crud
from shared.db.datasets.models import Dataset
from shared.db.datasets.schemas import DatasetCreate, DatasetRead


router = APIRouter(prefix="/datasets")


def dataset_response(item: Dataset, files: int) -> DatasetRead:
    return DatasetRead(id=item.id, name=item.name, files=files)


@router.get("", response_model=list[DatasetRead])
async def list_datasets() -> list[DatasetRead]:
    with database_session() as session:
        _ensure_synthesis_dataset(session)
        rows = dataset_crud.list_dataset_file_counts(session)
        return [dataset_response(item, files) for item, files in rows]


def _ensure_synthesis_dataset(session) -> None:
    if any(item.name == "synthesis" for item in dataset_crud.list_datasets(session)):
        return
    dataset_crud.create_dataset(session, DatasetCreate(name="synthesis"))


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(request: DatasetCreate) -> DatasetRead:
    try:
        with database_session() as session:
            item = dataset_crud.create_dataset(session, request)
            return dataset_response(item, 0)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str) -> None:
    try:
        with database_session() as session:
            dataset_crud.delete_dataset(session, uuid.UUID(dataset_id))
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
