from collections.abc import Sequence
import uuid
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.file_store import (
    checkpoint_cache_path,
    checkpoint_tar,
    extra_file_cache_path,
    object_store,
    stored_bytes,
    stored_path,
)
from shared.db.assets.models import BucketFile, Checkpoint, Config, ExtraFile
from shared.db.assets.schemas import (
    BucketFileCreate,
    CheckpointCreate,
    CheckpointUpdate,
    ConfigCreate,
    ExtraFileCreate,
    ExtraFilePathCreate,
    ExtraFileUpdate,
)
from shared.db.common import many, one
from shared.db.settings import crud as settings_crud
from shared.storage import ObjectStore


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
    _object_store(session).upload(path, stored.data)
    item = Checkpoint(
        id=item_id,
        name=payload.name,
        path=path,
        size=stored.size,
        content_hash=stored.content_hash,
        type_=payload.type_,
        metadata_=payload.metadata,
        job_id=payload.job_id,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def get_checkpoint(session: Session, checkpoint_id: UUID) -> Checkpoint:
    return one(session, Checkpoint, checkpoint_id)


def list_checkpoints(session: Session) -> Sequence[Checkpoint]:
    return many(session, Checkpoint)


def list_extra_files(session: Session, type_: str | None = None) -> Sequence[ExtraFile]:
    statement = select(ExtraFile)
    if type_ is not None:
        statement = statement.where(ExtraFile.type_ == type_)
    return session.execute(statement).scalars().all()


def list_configs(session: Session, type_: str | None = None) -> Sequence[Config]:
    statement = select(Config)
    if type_ is not None:
        statement = statement.where(Config.type_ == type_)
    return session.execute(statement).scalars().all()


def read_checkpoint(session: Session, checkpoint_id: UUID) -> bytes:
    item = one(session, Checkpoint, checkpoint_id)
    return _object_store(session).download(item.path)


def get_checkpoint_path(session: Session, checkpoint_id: UUID) -> Path:
    return checkpoint_cache_path(
        one(session, Checkpoint, checkpoint_id), _object_store(session)
    )


def update_checkpoint(
    session: Session, checkpoint_id: UUID, payload: CheckpointUpdate
) -> Checkpoint:
    item = one(session, Checkpoint, checkpoint_id)
    item.name = payload.name
    item.type_ = payload.type_
    item.metadata_ = payload.metadata
    item.job_id = payload.job_id
    if payload.folder_path is not None:
        stored = checkpoint_tar(payload.folder_path)
        _object_store(session).upload(item.path, stored.data)
        item.size = stored.size
        item.content_hash = stored.content_hash
    session.commit()
    session.refresh(item)
    return item


def delete_checkpoint(session: Session, checkpoint_id: UUID) -> None:
    item = one(session, Checkpoint, checkpoint_id)
    _object_store(session).delete(item.path)
    session.delete(item)
    session.commit()


def create_extra_file(session: Session, payload: ExtraFileCreate) -> ExtraFile:
    return bulk_create_extra_files(session, [payload])[0]


def bulk_create_extra_files(
    session: Session,
    payloads: Sequence[ExtraFileCreate],
    store: ObjectStore | None = None,
) -> list[ExtraFile]:
    if not payloads:
        return []
    resolved_store = store or _object_store(session)
    writes = []
    for payload in payloads:
        stored = stored_bytes(payload.data)
        item_id = uuid.uuid4()
        path = f"extra-files/{item_id}"
        item = ExtraFile(
            id=item_id,
            name=payload.name,
            path=path,
            size=stored.size,
            content_hash=stored.content_hash,
            type_=payload.type_,
            metadata_=payload.metadata,
        )
        writes.append((item, stored.data))
    uploaded_paths = []
    try:
        for item, data in writes:
            resolved_store.upload(item.path, data)
            uploaded_paths.append(item.path)
        items = [item for item, _ in writes]
        session.add_all(items)
        session.commit()
        return items
    except Exception:
        session.rollback()
        for path in uploaded_paths:
            resolved_store.delete(path)
        raise


def create_extra_file_from_path(
    session: Session,
    payload: ExtraFilePathCreate,
    store: ObjectStore | None = None,
) -> ExtraFile:
    return bulk_create_extra_files_from_paths(session, [payload], store)[0]


def bulk_create_extra_files_from_paths(
    session: Session,
    payloads: Sequence[ExtraFilePathCreate],
    store: ObjectStore | None = None,
) -> list[ExtraFile]:
    if not payloads:
        return []
    resolved_store = store or _object_store(session)
    writes = []
    for payload in payloads:
        stored = stored_path(payload.path)
        item_id = uuid.uuid4()
        item = ExtraFile(
            id=item_id,
            name=payload.name,
            path=f"extra-files/{item_id}",
            size=stored.size,
            content_hash=stored.content_hash,
            type_=payload.type_,
            metadata_=payload.metadata,
        )
        writes.append((item, stored.path))
    uploaded_paths = []
    try:
        for item, source in writes:
            resolved_store.upload_path(item.path, source)
            uploaded_paths.append(item.path)
        items = [item for item, _source in writes]
        session.add_all(items)
        session.commit()
        return items
    except Exception:
        session.rollback()
        for path in uploaded_paths:
            resolved_store.delete(path)
        raise


def get_extra_file(session: Session, extra_file_id: UUID) -> ExtraFile:
    return one(session, ExtraFile, extra_file_id)


def read_extra_file(session: Session, extra_file_id: UUID) -> bytes:
    return get_extra_file_path(session, extra_file_id).read_bytes()


def get_extra_file_path(session: Session, extra_file_id: UUID) -> Path:
    return extra_file_cache_path(
        one(session, ExtraFile, extra_file_id), _object_store(session)
    )


def update_extra_file(
    session: Session, extra_file_id: UUID, payload: ExtraFileUpdate
) -> ExtraFile:
    item = one(session, ExtraFile, extra_file_id)
    item.name = payload.name
    item.type_ = payload.type_
    item.metadata_ = payload.metadata
    if payload.data is not None:
        stored = stored_bytes(payload.data)
        _object_store(session).upload(item.path, stored.data)
        item.size = stored.size
        item.content_hash = stored.content_hash
    session.commit()
    session.refresh(item)
    return item


def delete_extra_file(session: Session, extra_file_id: UUID) -> None:
    item = one(session, ExtraFile, extra_file_id)
    _object_store(session).delete(item.path)
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


def _object_store(session: Session) -> ObjectStore:
    return object_store(settings_crud.object_store_config(session))
