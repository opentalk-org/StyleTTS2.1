from fastapi import APIRouter, HTTPException

from shared.db import database_session
from shared.db.settings import crud
from shared.db.settings.schemas import (
    IntegrationSettingsPayload,
    IntegrationSettingsRead,
    StorageSettingsPayload,
    StorageSettingsRead,
)
from shared.storage import ObjectStoreConfig, S3ObjectStore

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/storage", response_model=StorageSettingsRead)
async def get_storage_settings() -> StorageSettingsRead:
    with database_session() as session:
        return StorageSettingsRead.model_validate(crud.get_storage_settings(session))


@router.put("/storage", response_model=StorageSettingsRead)
async def update_storage_settings(payload: StorageSettingsPayload) -> StorageSettingsRead:
    with database_session() as session:
        return StorageSettingsRead.model_validate(crud.update_storage_settings(session, payload))


@router.post("/storage/test")
async def test_storage_settings(payload: StorageSettingsPayload) -> dict[str, bool]:
    try:
        S3ObjectStore(_object_store_config(payload)).test_connection()
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"ok": True}


@router.get("/integrations", response_model=IntegrationSettingsRead)
async def get_integration_settings() -> IntegrationSettingsRead:
    with database_session() as session:
        return IntegrationSettingsRead.model_validate(crud.get_integration_settings(session))


@router.put("/integrations", response_model=IntegrationSettingsRead)
async def update_integration_settings(payload: IntegrationSettingsPayload) -> IntegrationSettingsRead:
    with database_session() as session:
        return IntegrationSettingsRead.model_validate(crud.update_integration_settings(session, payload))


def _object_store_config(payload: StorageSettingsPayload) -> ObjectStoreConfig:
    return ObjectStoreConfig(
        bucket=payload.bucket,
        folder=payload.folder,
        endpoint_url=payload.endpoint_url,
        region_name=payload.region_name,
        access_key_id=payload.access_key_id,
        secret_access_key=payload.secret_access_key,
    )
