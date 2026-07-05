from fastapi import APIRouter

from shared.db import database_session
from shared.db.settings import crud
from shared.db.settings.schemas import StorageSettingsPayload, StorageSettingsRead

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/storage", response_model=StorageSettingsRead)
async def get_storage_settings() -> StorageSettingsRead:
    with database_session() as session:
        return StorageSettingsRead.model_validate(crud.get_storage_settings(session))


@router.put("/storage", response_model=StorageSettingsRead)
async def update_storage_settings(payload: StorageSettingsPayload) -> StorageSettingsRead:
    with database_session() as session:
        return StorageSettingsRead.model_validate(crud.update_storage_settings(session, payload))
