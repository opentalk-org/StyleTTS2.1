from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RunnerCreate(BaseModel):
    name: str
    hostname: str
    port: int


class RunnerRead(RunnerCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
