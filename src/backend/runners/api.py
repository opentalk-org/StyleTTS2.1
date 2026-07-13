from fastapi import APIRouter, status

from backend.runners.schemas import RunnerPage, RunnerRegisterRequest
from backend.runners.schemas import RunnerStatusRead
from backend.runners.service import runner_is_online
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
    online = runner_is_online(row.last_seen_at)
    return RunnerStatusRead(
        id=row.id,
        name=row.name,
        hostname=row.hostname,
        port=row.port,
        gpu_index=row.gpu_index,
        capabilities=row.capabilities,
        online=online,
        stale=row.last_seen_at is not None and not online,
        busy=bool(row.active_run_ids),
        active_run_ids=row.active_run_ids,
        process_id=row.process_id,
        last_seen_at=row.last_seen_at,
    )
