from collections.abc import Sequence
import uuid
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile, Checkpoint, Config, ExtraFile
from shared.db.assets.schemas import BucketFileCreate, ConfigCreate, FileAssetCreate, FileAssetUpdate
from shared.db.common import many, one
from shared.storage import ObjectStoreConfig, S3ObjectStore
from shared.storage.object_store import S3ObjectStore as ObjectStore


def list_bucket_files(session: Session) -> Sequence[BucketFile]:
    return many(session, BucketFile)


def get_bucket_file(session: Session, bucket_file_id: UUID) -> BucketFile:
    return one(session, BucketFile, bucket_file_id)


def create_bucket_file(session: Session, payload: BucketFileCreate) -> BucketFile:
    item = BucketFile(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def create_checkpoint(session: Session, payload: FileAssetCreate) -> Checkpoint:
    return _create_file_asset(session, Checkpoint, "checkpoints", payload)


def get_checkpoint(session: Session, checkpoint_id: UUID) -> Checkpoint:
    return one(session, Checkpoint, checkpoint_id)


def read_checkpoint(session: Session, checkpoint_id: UUID) -> bytes:
    item = one(session, Checkpoint, checkpoint_id)
    return _object_store().download(item.path)


def update_checkpoint(session: Session, checkpoint_id: UUID, payload: FileAssetUpdate) -> Checkpoint:
    item = one(session, Checkpoint, checkpoint_id)
    _update_file_asset(item, payload)
    session.commit()
    session.refresh(item)
    return item


def delete_checkpoint(session: Session, checkpoint_id: UUID) -> None:
    item = one(session, Checkpoint, checkpoint_id)
    _object_store().delete(item.path)
    session.delete(item)
    session.commit()


def create_extra_file(session: Session, payload: FileAssetCreate) -> ExtraFile:
    return _create_file_asset(session, ExtraFile, "extra-files", payload)


def get_extra_file(session: Session, extra_file_id: UUID) -> ExtraFile:
    return one(session, ExtraFile, extra_file_id)


def read_extra_file(session: Session, extra_file_id: UUID) -> bytes:
    item = one(session, ExtraFile, extra_file_id)
    return _object_store().download(item.path)


def update_extra_file(session: Session, extra_file_id: UUID, payload: FileAssetUpdate) -> ExtraFile:
    item = one(session, ExtraFile, extra_file_id)
    _update_file_asset(item, payload)
    session.commit()
    session.refresh(item)
    return item


def delete_extra_file(session: Session, extra_file_id: UUID) -> None:
    item = one(session, ExtraFile, extra_file_id)
    _object_store().delete(item.path)
    session.delete(item)
    session.commit()


def create_config(session: Session, payload: ConfigCreate) -> Config:
    data = payload.model_dump()
    data["metadata_"] = data.pop("metadata")
    item = Config(**data)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def _create_file_asset(
    session: Session,
    model: type[Checkpoint] | type[ExtraFile],
    path_prefix: str,
    payload: FileAssetCreate,
) -> Checkpoint | ExtraFile:
    item_id = uuid.uuid4()
    path = f"{path_prefix}/{item_id}"
    _object_store().upload(path, payload.data)
    item = model(
        id=item_id,
        name=payload.name,
        path=path,
        size=len(payload.data),
        type_=payload.type_,
        metadata_=payload.metadata,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def _update_file_asset(item: Checkpoint | ExtraFile, payload: FileAssetUpdate) -> None:
    item.name = payload.name
    item.type_ = payload.type_
    item.metadata_ = payload.metadata
    if payload.data is not None:
        _object_store().upload(item.path, payload.data)
        item.size = len(payload.data)


def _object_store() -> ObjectStore:
    return S3ObjectStore(ObjectStoreConfig.from_env())
