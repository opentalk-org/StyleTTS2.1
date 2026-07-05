from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class WorkflowCreate(BaseModel):
    name: str
    data: dict[str, Any]
    hidden: bool


class WorkflowRead(WorkflowCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
