from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AssetKind(str, Enum):
    CHECKPOINT = "checkpoint"
    FILE = "file"


class BucketKind(str, Enum):
    AUDIO = "audio"
    WAVEFORM = "waveform"


class BucketFileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    kind: BucketKind
    path: str
    size: int


class AssetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    kind: AssetKind
    name: str
    path: str
    size: int
    content_hash: str
    type: str
    metadata: dict[str, Any]
    run_id: UUID | None


class ConfigRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    type: str
    metadata: dict[str, Any]
