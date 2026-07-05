from uuid import UUID

from pydantic import BaseModel

from shared.db.workflows.schemas import WorkflowCreate, WorkflowRead
from shared.schemas import InlineGraphRunRequest, RunStatus


class WorkflowCompileRequest(BaseModel):
    run_id: str | None = None


class WorkflowCompileResponse(BaseModel):
    workflow_id: UUID
    request: InlineGraphRunRequest


class WorkflowStartResponse(BaseModel):
    workflow: WorkflowRead
    run: RunStatus


WorkflowSaveRequest = WorkflowCreate
