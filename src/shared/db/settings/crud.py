from sqlalchemy.orm import Session

from shared.db.settings.models import StorageSettings
from shared.db.settings.schemas import StorageSettingsPayload
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
    item.endpoint_url = payload.endpoint_url
    item.region_name = payload.region_name
    item.access_key_id = payload.access_key_id
    item.secret_access_key = payload.secret_access_key
    session.commit()
    session.refresh(item)
    return item


def object_store_config(session: Session) -> ObjectStoreConfig:
    item = get_storage_settings(session)
    return ObjectStoreConfig(
        bucket=item.bucket,
        endpoint_url=item.endpoint_url,
        region_name=item.region_name,
        access_key_id=item.access_key_id,
        secret_access_key=item.secret_access_key,
    )
