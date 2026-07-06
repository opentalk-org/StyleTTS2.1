from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DatasetCreate(BaseModel):
    name: str


class DatasetRead(DatasetCreate):
    id: UUID
    files: int
    model_config = ConfigDict(from_attributes=True)
