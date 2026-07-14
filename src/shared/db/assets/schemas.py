from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BucketFileCreate(BaseModel):
    path: str
    size: int
    used_bytes: int
    sealed: bool


class BucketFileRead(BucketFileCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)


class CheckpointCreate(BaseModel):
    name: str
    folder_path: Path
    type_: str
    metadata: dict[str, Any]
    job_id: str | None = None


class CheckpointUpdate(BaseModel):
    name: str
    folder_path: Path | None
    type_: str
    metadata: dict[str, Any]
    job_id: str | None = None


class CheckpointRead(BaseModel):
    id: UUID
    name: str
    path: str
    size: int
    content_hash: str
    type_: str
    metadata: dict[str, Any] = Field(alias="metadata_")
    job_id: str | None
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ExtraFileCreate(BaseModel):
    name: str
    data: bytes
    type_: str
    metadata: dict[str, Any]


class ExtraFilePathCreate(BaseModel):
    name: str
    path: Path
    type_: str
    metadata: dict[str, Any]


class ExtraFileUpdate(BaseModel):
    name: str
    data: bytes | None
    type_: str
    metadata: dict[str, Any]


FileAssetCreate = ExtraFileCreate
FileAssetUpdate = ExtraFileUpdate


class FileAssetRead(BaseModel):
    id: UUID
    name: str
    path: str
    size: int
    content_hash: str
    type_: str
    metadata: dict[str, Any] = Field(alias="metadata_")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ConfigCreate(BaseModel):
    name: str
    type_: str
    metadata: dict[str, Any]


class ConfigRead(ConfigCreate):
    id: UUID
    metadata: dict[str, Any] = Field(alias="metadata_")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
