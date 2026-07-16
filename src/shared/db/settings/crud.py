from sqlalchemy.orm import Session

from shared.db.settings.models import IntegrationSettings, StorageSettings
from shared.db.settings.schemas import IntegrationSettingsPayload, StorageSettingsPayload
from shared.storage import ObjectStoreConfig


def get_storage_settings(session: Session) -> StorageSettings:
    item = session.query(StorageSettings).order_by(StorageSettings.id).first()
    if item is not None:
        return item
    item = StorageSettings(**StorageSettingsPayload().model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def update_storage_settings(session: Session, payload: StorageSettingsPayload) -> StorageSettings:
    item = get_storage_settings(session)
    item.bucket = payload.bucket
    item.folder = payload.folder
    item.endpoint_url = payload.endpoint_url
    item.region_name = payload.region_name
    item.access_key_id = payload.access_key_id
    item.secret_access_key = payload.secret_access_key
    session.commit()
    session.refresh(item)
    return item


def get_integration_settings(session: Session) -> IntegrationSettings:
    item = session.query(IntegrationSettings).order_by(IntegrationSettings.id).first()
    if item is not None:
        return item
    item = IntegrationSettings(**IntegrationSettingsPayload().model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def update_integration_settings(session: Session, payload: IntegrationSettingsPayload) -> IntegrationSettings:
    item = get_integration_settings(session)
    item.hf_token = payload.hf_token
    item.openrouter_token = payload.openrouter_token
    item.wandb_url = payload.wandb_url
    session.commit()
    session.refresh(item)
    return item


def object_store_config(session: Session) -> ObjectStoreConfig:
    item = get_storage_settings(session)
    return ObjectStoreConfig(
        bucket=item.bucket,
        folder=item.folder,
        endpoint_url=item.endpoint_url,
        region_name=item.region_name,
        access_key_id=item.access_key_id,
        secret_access_key=item.secret_access_key,
    )
