from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from backend.service import BackendManager, DuplicateRunError
from backend.workflows.schemas import WorkflowCompileRequest, WorkflowCompileResponse, WorkflowSaveRequest, WorkflowStartResponse
from backend.workflows.service import compile_workflow_definition, load_example_workflows
from shared.db import database_session
from shared.db.workflows import crud
from shared.db.workflows.schemas import WorkflowRead


def workflow_router(manager: BackendManager) -> APIRouter:
    router = APIRouter(prefix="/workflows", tags=["workflows"])

    @router.get("", response_model=list[WorkflowRead])
    async def list_workflows() -> list[WorkflowRead]:
        with database_session() as session:
            return [WorkflowRead.model_validate(item) for item in crud.list_workflows(session)]

    @router.post("", response_model=WorkflowRead, status_code=status.HTTP_201_CREATED)
    async def create_workflow(payload: WorkflowSaveRequest) -> WorkflowRead:
        with database_session() as session:
            return WorkflowRead.model_validate(crud.create_workflow(session, payload))

    # Declared before "/{workflow_id}" so "examples" is not parsed as a UUID.
    @router.get("/examples", response_model=list[WorkflowRead])
    async def list_example_workflows() -> list[WorkflowRead]:
        return load_example_workflows()

    @router.get("/{workflow_id}", response_model=WorkflowRead)
    async def get_workflow(workflow_id: UUID) -> WorkflowRead:
        try:
            with database_session() as session:
                return WorkflowRead.model_validate(crud.get_workflow(session, workflow_id))
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_workflow(workflow_id: UUID) -> None:
        try:
            with database_session() as session:
                crud.delete_workflow(session, workflow_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @router.post("/{workflow_id}/compile", response_model=WorkflowCompileResponse)
    async def compile_workflow(workflow_id: UUID, payload: WorkflowCompileRequest | None = None) -> WorkflowCompileResponse:
        workflow = await get_workflow(workflow_id)
        request = compile_workflow_definition(workflow.data, workflow_id, payload.run_id if payload else None)
        return WorkflowCompileResponse(workflow_id=workflow_id, request=request)

    @router.post("/{workflow_id}/runs", response_model=WorkflowStartResponse, status_code=status.HTTP_202_ACCEPTED)
    async def start_workflow(workflow_id: UUID) -> WorkflowStartResponse:
        workflow = await get_workflow(workflow_id)
        try:
            request = compile_workflow_definition(workflow.data, workflow_id)
            run = await manager.start_inline_graph(request, name=workflow.name)
            return WorkflowStartResponse(workflow=workflow, run=run)
        except DuplicateRunError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    return router
