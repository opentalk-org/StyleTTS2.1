from fastapi import APIRouter, HTTPException, status
from uuid import UUID

from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointRead, CheckpointUpdate


router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])


@router.get("", response_model=list[CheckpointRead])
async def list_checkpoints() -> list[CheckpointRead]:
    with database_session() as session:
        return [CheckpointRead.model_validate(item) for item in asset_crud.list_checkpoints(session)]


@router.patch("/{checkpoint_id}", response_model=CheckpointRead)
async def update_checkpoint(checkpoint_id: UUID, payload: CheckpointUpdate) -> CheckpointRead:
    try:
        with database_session() as session:
            return CheckpointRead.model_validate(asset_crud.update_checkpoint(session, checkpoint_id, payload))
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.delete("/{checkpoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_checkpoint(checkpoint_id: UUID) -> None:
    try:
        with database_session() as session:
            asset_crud.delete_checkpoint(session, checkpoint_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
