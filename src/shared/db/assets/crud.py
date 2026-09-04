from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from shared.db.assets import clickhouse as ch
from shared.db.assets.clickhouse import (
    AssetKind,
    AssetRecord,
    BucketFileRecord,
    BucketKind,
    ConfigRecord,
)
from shared.db.assets.file_store import (
    checkpoint_cache_path,
    checkpoint_tar,
    extra_file_cache_path,
    populate_checkpoint_cache,
    stored_bytes,
    stored_path,
)
from shared.db.assets.schemas import (
    BucketFileCreate,
    CheckpointCreate,
    CheckpointUpdate,
    ConfigCreate,
    ConfigUpdate,
    ExtraFileCreate,
    ExtraFilePathCreate,
    ExtraFileUpdate,
)
from shared.db.settings import crud as settings_crud


def list_bucket_files() -> Sequence[BucketFileRecord]:
    return ch.list_bucket_files()


def get_bucket_file(bucket_file_id: UUID) -> BucketFileRecord:
    return ch.get_bucket_file(bucket_file_id)


def create_bucket_file(
    payload: BucketFileCreate,
) -> BucketFileRecord:
    item = BucketFileRecord(
        id=uuid4(), kind=BucketKind.AUDIO, path=payload.path, size=payload.size
    )
    ch.create_bucket_files([item])
    return item


def delete_unreferenced_bucket_files(
    session: Session, bucket_file_ids: Sequence[UUID]
) -> int:
    candidates = ch.get_bucket_files(bucket_file_ids)
    referenced = ch.referenced_bucket_file_ids([item.id for item in candidates])
    garbage = [item for item in candidates if item.id not in referenced]
    store = settings_crud.object_store(session)
    for item in garbage:
        store.delete(item.path)
    ch.delete_bucket_files([item.id for item in garbage])
    return len(garbage)


def create_checkpoint(session: Session, payload: CheckpointCreate) -> AssetRecord:
    item_id = uuid4()
    path = f"checkpoints/{item_id}.tar"
    with checkpoint_tar(payload.folder_path) as stored:
        settings_crud.object_store(session).upload_path(path, stored.path)
        item = AssetRecord(
            id=item_id,
            updated_at=datetime.now(UTC),
            kind=AssetKind.CHECKPOINT,
            name=payload.name,
            path=path,
            size=stored.size,
            content_hash=stored.content_hash,
            type=payload.type_,
            metadata=payload.metadata,
            run_id=UUID(payload.job_id) if payload.job_id else None,
        )
    ch.create_assets([item])
    populate_checkpoint_cache(payload.folder_path, item.content_hash)
    return item


def get_checkpoint(checkpoint_id: UUID) -> AssetRecord:
    item = ch.get_asset(checkpoint_id)
    if item.kind != AssetKind.CHECKPOINT:
        raise KeyError(f"Checkpoint not found: {checkpoint_id}")
    return item


def list_checkpoints() -> Sequence[AssetRecord]:
    return ch.list_assets(AssetKind.CHECKPOINT)


def list_extra_files(
    type_: str | None = None,
) -> Sequence[AssetRecord]:
    return ch.list_assets(AssetKind.FILE, type_)


def list_configs(type_: str | None = None) -> Sequence[ConfigRecord]:
    return ch.list_configs(type_)


def update_config(
    config_id: UUID, payload: ConfigUpdate
) -> ConfigRecord:
    item = ch.get_config(config_id).model_copy(
        update={
            "updated_at": datetime.now(UTC),
            "name": payload.name,
            "type": payload.type_,
            "metadata": payload.metadata,
        }
    )
    return ch.update_config(item)


def read_checkpoint(session: Session, checkpoint_id: UUID) -> bytes:
    return settings_crud.object_store(session).download(
        get_checkpoint(checkpoint_id).path
    )


def get_checkpoint_path(session: Session, checkpoint_id: UUID) -> Path:
    return checkpoint_cache_path(
        get_checkpoint(checkpoint_id), settings_crud.object_store(session)
    )


def update_checkpoint(
    session: Session, checkpoint_id: UUID, payload: CheckpointUpdate
) -> AssetRecord:
    item = get_checkpoint(checkpoint_id)
    changes = {
        "updated_at": datetime.now(UTC),
        "name": payload.name,
        "type": payload.type_,
        "metadata": payload.metadata,
        "run_id": UUID(payload.job_id) if payload.job_id else None,
    }
    if payload.folder_path is not None:
        with checkpoint_tar(payload.folder_path) as stored:
            settings_crud.object_store(session).upload_path(item.path, stored.path)
            changes.update(size=stored.size, content_hash=stored.content_hash)
        populate_checkpoint_cache(payload.folder_path, stored.content_hash)
    return ch.update_asset(item.model_copy(update=changes))


def delete_checkpoint(session: Session, checkpoint_id: UUID) -> None:
    item = get_checkpoint(checkpoint_id)
    settings_crud.object_store(session).delete(item.path)
    ch.delete_asset(item.id)


def create_extra_file(session: Session, payload: ExtraFileCreate) -> AssetRecord:
    return bulk_create_extra_files(session, [payload])[0]


def bulk_create_extra_files(
    session: Session, payloads: Sequence[ExtraFileCreate]
) -> list[AssetRecord]:
    store = settings_crud.object_store(session)
    items = []
    for payload in payloads:
        stored = stored_bytes(payload.data)
        item = _file_record(
            payload.name,
            payload.type_,
            payload.metadata,
            stored.size,
            stored.content_hash,
        )
        store.upload(item.path, stored.data)
        items.append(item)
    ch.create_assets(items)
    return items


def create_extra_file_from_path(
    session: Session, payload: ExtraFilePathCreate
) -> AssetRecord:
    return bulk_create_extra_files_from_paths(session, [payload])[0]


def bulk_create_extra_files_from_paths(
    session: Session, payloads: Sequence[ExtraFilePathCreate]
) -> list[AssetRecord]:
    store = settings_crud.object_store(session)
    items = []
    for payload in payloads:
        stored = stored_path(payload.path)
        item = _file_record(
            payload.name,
            payload.type_,
            payload.metadata,
            stored.size,
            stored.content_hash,
        )
        store.upload_path(item.path, stored.path)
        items.append(item)
    ch.create_assets(items)
    return items


def get_extra_file(extra_file_id: UUID) -> AssetRecord:
    item = ch.get_asset(extra_file_id)
    if item.kind != AssetKind.FILE:
        raise KeyError(f"File asset not found: {extra_file_id}")
    return item


def read_extra_file(session: Session, extra_file_id: UUID) -> bytes:
    return get_extra_file_path(session, extra_file_id).read_bytes()


def get_extra_file_path(session: Session, extra_file_id: UUID) -> Path:
    return extra_file_cache_path(
        get_extra_file(extra_file_id), settings_crud.object_store(session)
    )


def update_extra_file(
    session: Session, extra_file_id: UUID, payload: ExtraFileUpdate
) -> AssetRecord:
    item = get_extra_file(extra_file_id)
    changes = {
        "updated_at": datetime.now(UTC),
        "name": payload.name,
        "type": payload.type_,
        "metadata": payload.metadata,
    }
    if payload.data is not None:
        stored = stored_bytes(payload.data)
        settings_crud.object_store(session).upload(item.path, stored.data)
        changes.update(size=stored.size, content_hash=stored.content_hash)
    return ch.update_asset(item.model_copy(update=changes))


def delete_extra_file(session: Session, extra_file_id: UUID) -> None:
    item = get_extra_file(extra_file_id)
    settings_crud.object_store(session).delete(item.path)
    ch.delete_asset(item.id)


def create_config(payload: ConfigCreate) -> ConfigRecord:
    return ch.create_config(
        ConfigRecord(
            id=uuid4(),
            updated_at=datetime.now(UTC),
            name=payload.name,
            type=payload.type_,
            metadata=payload.metadata,
        )
    )


def _file_record(
    name: str, type_: str, metadata: dict, size: int, content_hash: str
) -> AssetRecord:
    item_id = uuid4()
    return AssetRecord(
        id=item_id,
        updated_at=datetime.now(UTC),
        kind=AssetKind.FILE,
        name=name,
        path=f"extra-files/{item_id}",
        size=size,
        content_hash=content_hash,
        type=type_,
        metadata=metadata,
        run_id=None,
    )
