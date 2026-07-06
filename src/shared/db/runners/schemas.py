from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RunnerCreate(BaseModel):
    name: str
    hostname: str
    port: int
    gpu_index: int | None = None


class RunnerRead(RunnerCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
