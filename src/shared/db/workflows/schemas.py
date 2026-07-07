from uuid import UUID

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas import GraphEdgeRequest, GraphNodeRequest, RunContextRequest


class WorkflowLaunchSource(BaseModel):
    kind: Literal["selected_audio", "dataset_audio", "all_audio"]
    audio_file_ids: list[UUID] = Field(default_factory=list)
    dataset_id: UUID | None = None
    include_virtual: bool = False


class ControlTarget(BaseModel):
    node_id: str
    key: str


class WorkflowControl(BaseModel):
    id: str
    label: str = ""
    description: str = ""
    targets: list[ControlTarget] = Field(default_factory=list)


class ControlPanel(BaseModel):
    id: str
    x: float = 0
    y: float = 0
    title: str = "Controls"
    controls: list[WorkflowControl] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    nodes: list[GraphNodeRequest]
    edges: list[GraphEdgeRequest] = Field(default_factory=list)
    context: RunContextRequest = Field(default_factory=RunContextRequest)
    launch_source: WorkflowLaunchSource | None = None
    panels: list[ControlPanel] = Field(default_factory=list)


class WorkflowCreate(BaseModel):
    name: str
    data: WorkflowDefinition
    hidden: bool


class WorkflowRead(WorkflowCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
