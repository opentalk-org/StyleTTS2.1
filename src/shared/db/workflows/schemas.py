from uuid import UUID

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas import GraphEdgeRequest, GraphNodeRequest, RunContextRequest


class WorkflowLaunchSource(BaseModel):
    kind: Literal["selected_audio", "dataset_audio", "all_audio"]
    audio_file_ids: list[UUID] = Field(default_factory=list)
    dataset_id: UUID | None = None
    include_virtual: bool = False


class WorkflowDefinition(BaseModel):
    nodes: list[GraphNodeRequest]
    edges: list[GraphEdgeRequest] = Field(default_factory=list)
    context: RunContextRequest = Field(default_factory=RunContextRequest)
    launch_source: WorkflowLaunchSource | None = None


class WorkflowCreate(BaseModel):
    name: str
    data: WorkflowDefinition
    hidden: bool


class WorkflowRead(WorkflowCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
