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


class FileAssetCreate(BaseModel):
    name: str
    data: bytes
    type_: str
    metadata: dict[str, Any]


class FileAssetUpdate(BaseModel):
    name: str
    data: bytes | None
    type_: str
    metadata: dict[str, Any]


class FileAssetRead(BaseModel):
    id: UUID
    name: str
    path: str
    size: int
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
