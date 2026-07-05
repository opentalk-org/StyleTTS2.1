from uuid import UUID

from pydantic import BaseModel, ConfigDict


class InitializationCreate(BaseModel):
    is_initialized: bool


class InitializationRead(InitializationCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
