from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.models import Checkpoint, ExtraFile
from shared.db.assets.schemas import CheckpointCreate, CheckpointUpdate, ExtraFileCreate, ExtraFileUpdate

from runner.nodes.assets.catalog_runtime.http import download_url_bytes, download_url_to_file
from runner.nodes.assets.catalog_runtime.types import CheckpointSpec, ExtraFileSpec


_LOGGER = logging.getLogger(__name__)


def ensure_checkpoint_bundle(spec: CheckpointSpec, *, logger: logging.Logger | None = None) -> tuple[Checkpoint, bool]:
    log = logger or _LOGGER
    with database_session() as session:
        existing = _find_checkpoint(session, spec.key)
        if existing is not None:
            path = asset_crud.get_checkpoint_path(session, existing.id)
            if spec.is_valid(path):
                log.info("catalog checkpoint cached key=%s name=%s (reusing existing)", spec.key, spec.name)
                metadata = _checkpoint_metadata(spec, path, existing.metadata_)
                updated = asset_crud.update_checkpoint(
                    session,
                    existing.id,
                    CheckpointUpdate(name=spec.name, folder_path=None, type_=spec.type_.value, metadata=metadata),
                )
                return updated, True
        log.info("catalog checkpoint download starting key=%s name=%s files=%s", spec.key, spec.name, len(spec.files))
        with TemporaryDirectory(prefix=f"runflow-catalog-{spec.key}-") as tmp:
            folder = Path(tmp)
            _download_checkpoint_files(spec, folder, log)
            if not spec.is_valid(folder):
                raise ValueError(f"{spec.key}_bundle_invalid_after_download")
            log.info("catalog checkpoint download complete key=%s name=%s", spec.key, spec.name)
            metadata = _checkpoint_metadata(spec, folder, {})
            if existing is None:
                created = asset_crud.create_checkpoint(
                    session,
                    CheckpointCreate(name=spec.name, folder_path=folder, type_=spec.type_.value, metadata=metadata),
                )
                return created, False
            updated = asset_crud.update_checkpoint(
                session,
                existing.id,
                CheckpointUpdate(name=spec.name, folder_path=folder, type_=spec.type_.value, metadata=metadata),
            )
            return updated, False


def ensure_extra_file(spec: ExtraFileSpec, *, logger: logging.Logger | None = None) -> tuple[ExtraFile, bool]:
    log = logger or _LOGGER
    with database_session() as session:
        existing = _find_extra_file(session, spec.key)
        metadata = {**spec.metadata, "catalog_key": spec.key}
        if existing is not None:
            path = asset_crud.get_extra_file_path(session, existing.id)
            if path.read_bytes().strip():
                log.info("catalog extra file cached key=%s name=%s (reusing existing)", spec.key, spec.name)
                updated = asset_crud.update_extra_file(
                    session,
                    existing.id,
                    ExtraFileUpdate(name=spec.name, data=None, type_=spec.type_.value, metadata=metadata),
                )
                return updated, True
        log.info("catalog extra file download starting key=%s name=%s", spec.key, spec.name)
        data = download_url_bytes(spec.url, error_prefix=f"{spec.key}_download_failed", logger=log)
        if not data.strip():
            raise ValueError(f"{spec.key}_download_empty")
        if existing is None:
            created = asset_crud.create_extra_file(
                session,
                ExtraFileCreate(name=spec.name, data=data, type_=spec.type_.value, metadata=metadata),
            )
            return created, False
        updated = asset_crud.update_extra_file(
            session,
            existing.id,
            ExtraFileUpdate(name=spec.name, data=data, type_=spec.type_.value, metadata=metadata),
        )
        return updated, False


def checkpoint_payload(item: Checkpoint, *, skipped: bool, filename: str | None = None) -> dict[str, Any]:
    payload = {
        "checkpoint_id": str(item.id),
        "name": item.name,
        "type": item.type_,
        "path": item.path,
        "size": item.size,
        "content_hash": item.content_hash,
        "metadata": item.metadata_,
        "skipped": skipped,
    }
    if filename is not None:
        payload["filename"] = filename
    return payload


def extra_file_payload(item: ExtraFile, *, skipped: bool) -> dict[str, Any]:
    return {
        "extra_file_id": str(item.id),
        "name": item.name,
        "type": item.type_,
        "path": item.path,
        "size": item.size,
        "content_hash": item.content_hash,
        "metadata": item.metadata_,
        "skipped": skipped,
    }


def _download_checkpoint_files(spec: CheckpointSpec, folder: Path, logger: logging.Logger) -> None:
    total = len(spec.files)
    for index, file in enumerate(spec.files, start=1):
        logger.info("catalog file download key=%s file=%s index=%s/%s", spec.key, file.name, index, total)
        download_url_to_file(
            file.url, folder / file.name, error_prefix=f"{spec.key}_download_failed", logger=logger
        )


def _checkpoint_metadata(spec: CheckpointSpec, folder: Path, current: dict[str, Any]) -> dict[str, Any]:
    return {**current, **spec.metadata, **spec.metadata_from_path(folder), "catalog_key": spec.key}


def _find_checkpoint(session, key: str) -> Checkpoint | None:
    for checkpoint in asset_crud.list_checkpoints(session):
        metadata = checkpoint.metadata_
        if "catalog_key" in metadata and metadata["catalog_key"] == key:
            return checkpoint
    return None


def _find_extra_file(session, key: str) -> ExtraFile | None:
    for extra_file in asset_crud.list_extra_files(session):
        metadata = extra_file.metadata_
        if "catalog_key" in metadata and metadata["catalog_key"] == key:
            return extra_file
    return None

