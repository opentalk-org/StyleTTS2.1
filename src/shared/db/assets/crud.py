from collections.abc import Sequence
import uuid
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.assets.file_store import (
    checkpoint_cache_path,
    checkpoint_tar,
    extra_file_cache_path,
    object_store,
    stored_bytes,
)
from shared.db.assets.models import BucketFile, Checkpoint, Config, ExtraFile
from shared.db.assets.schemas import (
    BucketFileCreate,
    CheckpointCreate,
    CheckpointUpdate,
    ConfigCreate,
    ExtraFileCreate,
    ExtraFileUpdate,
)
from shared.db.common import many, one


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


def create_checkpoint(session: Session, payload: CheckpointCreate) -> Checkpoint:
    item_id = uuid.uuid4()
    path = f"checkpoints/{item_id}.tar"
    stored = checkpoint_tar(payload.folder_path)
    object_store().upload(path, stored.data)
    item = Checkpoint(
        id=item_id,
        name=payload.name,
        path=path,
        size=stored.size,
        content_hash=stored.content_hash,
        type_=payload.type_,
        metadata_=payload.metadata,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def get_checkpoint(session: Session, checkpoint_id: UUID) -> Checkpoint:
    return one(session, Checkpoint, checkpoint_id)


def read_checkpoint(session: Session, checkpoint_id: UUID) -> bytes:
    item = one(session, Checkpoint, checkpoint_id)
    return object_store().download(item.path)


def get_checkpoint_path(session: Session, checkpoint_id: UUID) -> Path:
    return checkpoint_cache_path(one(session, Checkpoint, checkpoint_id))


def update_checkpoint(session: Session, checkpoint_id: UUID, payload: CheckpointUpdate) -> Checkpoint:
    item = one(session, Checkpoint, checkpoint_id)
    item.name = payload.name
    item.type_ = payload.type_
    item.metadata_ = payload.metadata
    if payload.folder_path is not None:
        stored = checkpoint_tar(payload.folder_path)
        object_store().upload(item.path, stored.data)
        item.size = stored.size
        item.content_hash = stored.content_hash
    session.commit()
    session.refresh(item)
    return item


def delete_checkpoint(session: Session, checkpoint_id: UUID) -> None:
    item = one(session, Checkpoint, checkpoint_id)
    object_store().delete(item.path)
    session.delete(item)
    session.commit()


def create_extra_file(session: Session, payload: ExtraFileCreate) -> ExtraFile:
    stored = stored_bytes(payload.data)
    item_id = uuid.uuid4()
    path = f"extra-files/{item_id}"
    object_store().upload(path, stored.data)
    item = ExtraFile(
        id=item_id,
        name=payload.name,
        path=path,
        size=stored.size,
        content_hash=stored.content_hash,
        type_=payload.type_,
        metadata_=payload.metadata,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def get_extra_file(session: Session, extra_file_id: UUID) -> ExtraFile:
    return one(session, ExtraFile, extra_file_id)


def read_extra_file(session: Session, extra_file_id: UUID) -> bytes:
    return get_extra_file_path(session, extra_file_id).read_bytes()


def get_extra_file_path(session: Session, extra_file_id: UUID) -> Path:
    return extra_file_cache_path(one(session, ExtraFile, extra_file_id))


def update_extra_file(session: Session, extra_file_id: UUID, payload: ExtraFileUpdate) -> ExtraFile:
    item = one(session, ExtraFile, extra_file_id)
    item.name = payload.name
    item.type_ = payload.type_
    item.metadata_ = payload.metadata
    if payload.data is not None:
        stored = stored_bytes(payload.data)
        object_store().upload(item.path, stored.data)
        item.size = stored.size
        item.content_hash = stored.content_hash
    session.commit()
    session.refresh(item)
    return item


def delete_extra_file(session: Session, extra_file_id: UUID) -> None:
    item = one(session, ExtraFile, extra_file_id)
    object_store().delete(item.path)
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
