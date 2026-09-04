import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.db.clickhouse.types import utc_datetime


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
    used_bytes: int


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

    _updated_at_utc = field_validator("updated_at")(utc_datetime)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @property
    def type_(self) -> str:
        return self.type

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.metadata

    @property
    def job_id(self) -> str | None:
        return str(self.run_id) if self.run_id is not None else None


class ConfigRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    type: str
    metadata: dict[str, Any]

    _updated_at_utc = field_validator("updated_at")(utc_datetime)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @property
    def type_(self) -> str:
        return self.type

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.metadata
