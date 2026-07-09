from fastapi import APIRouter, HTTPException, Query, status

from shared.db import database_session
from shared.db.jobs import crud as jobs_crud
from shared.db.jobs.schemas import JobPage, JobRead, JobSummary
from shared.schemas import InlineGraphRunRequest

# Job removal is handled by the backend manager (`DELETE /jobs/{run_id}` in
# backend.api) so a still-running job can be stopped before its record is pruned.


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobPage)
async def list_jobs(limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)) -> JobPage:
    with database_session() as session:
        rows, total = jobs_crud.list_jobs(session, limit, offset)
        return JobPage(rows=[JobSummary.model_validate(item) for item in rows], total=total)


@router.get("/{run_id}", response_model=JobRead)
async def get_job(run_id: str) -> JobRead:
    try:
        with database_session() as session:
            return JobRead.model_validate(jobs_crud.get_job(session, run_id))
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/{run_id}/graph", response_model=InlineGraphRunRequest)
async def get_job_graph(run_id: str) -> InlineGraphRunRequest:
    try:
        with database_session() as session:
            item = jobs_crud.get_job(session, run_id)
            return InlineGraphRunRequest.model_validate(item.graph_request)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
