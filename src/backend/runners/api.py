from fastapi import APIRouter, status

from backend.runners.schemas import RunnerPage, RunnerRegisterRequest
from backend.runners.schemas import RunnerStatusRead
from backend.runners.service import runner_live_registry
from shared.db import database_session
from shared.db.runners import crud
from shared.db.runners.schemas import RunnerRead

router = APIRouter(prefix="/runners", tags=["runners"])


@router.get("", response_model=RunnerPage)
async def list_runners() -> RunnerPage:
    with database_session() as session:
        rows = [runner_response(RunnerRead.model_validate(item)) for item in crud.list_runners(session)]
        return RunnerPage(rows=rows, total=len(rows))


@router.post("", response_model=RunnerRead, status_code=status.HTTP_201_CREATED)
async def create_runner(payload: RunnerRegisterRequest) -> RunnerRead:
    with database_session() as session:
        return RunnerRead.model_validate(crud.create_runner(session, payload))


def runner_response(row: RunnerRead) -> RunnerStatusRead:
    heartbeat = runner_live_registry.heartbeat(row.name)
    online = runner_live_registry.is_online(heartbeat)
    active_run_ids = heartbeat.active_run_ids if heartbeat is not None else []
    return RunnerStatusRead(
        id=row.id,
        name=row.name,
        hostname=heartbeat.hostname if heartbeat is not None else row.hostname,
        port=heartbeat.port if heartbeat is not None else row.port,
        gpu_index=heartbeat.gpu_index if heartbeat is not None else row.gpu_index,
        resources=heartbeat.resources if heartbeat is not None else row.resources,
        online=online,
        stale=heartbeat is not None and not online,
        busy=bool(active_run_ids),
        active_run_ids=active_run_ids,
        process_id=heartbeat.process_id if heartbeat is not None else None,
        last_seen_at=heartbeat.created_at if heartbeat is not None else None,
    )
