from shared.db import database_session
from shared.db.jobs import crud as jobs_crud
from shared.db.jobs.schemas import JobUpsert, NodeLogUpsert
from shared.schemas import NodeLogResponseMessage


def persist_job(record) -> None:
    if record.graph_request is None:
        return
    payload = JobUpsert(
        run_id=record.run_id,
        name=record.name,
        state=record.state.value,
        graph_request=record.graph_request.model_dump(mode="json"),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
    )
    with database_session() as session:
        jobs_crud.upsert_job(session, payload)


def persist_node_log(response: NodeLogResponseMessage) -> None:
    payload = NodeLogUpsert(
        run_id=response.run_id,
        node_id=response.node_id,
        content=response.content,
        truncated=response.truncated,
        error=response.error,
    )
    with database_session() as session:
        jobs_crud.upsert_node_log(session, payload)
