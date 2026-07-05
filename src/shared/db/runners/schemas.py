from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunnerCreate(BaseModel):
    name: str
    hostname: str
    port: int
    gpu_index: int | None = None
    resources: dict[str, float] = Field(default_factory=dict)


class RunnerRead(RunnerCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
