from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session, defer

from shared.db.jobs.models import Job, NodeLog
from shared.db.jobs.schemas import JobUpsert, NodeLogUpsert
from shared.schemas import RunState


ACTIVE_JOB_STATES = {RunState.RUNNING.value, RunState.STOPPING.value}


class ActiveJobError(ValueError):
    pass


def list_jobs(session: Session, limit: int, offset: int) -> tuple[Sequence[Job], int]:
    # Defer the heavy JSONB columns; the list only needs summary fields, and the graph /
    # snapshot are fetched per run on demand.
    rows = session.execute(
        select(Job)
        .options(defer(Job.graph_request), defer(Job.snapshot))
        .order_by(desc(Job.updated_at))
        .limit(limit)
        .offset(offset)
    ).scalars().all()
    total = session.execute(select(func.count()).select_from(Job)).scalar_one()
    return rows, total


def list_all_job_ids(session: Session) -> list[str]:
    return list(session.execute(select(Job.run_id)).scalars().all())


def get_job(session: Session, run_id: str) -> Job:
    item = session.get(Job, run_id)
    if item is None:
        raise KeyError(f"Job not found: {run_id}")
    return item


def upsert_job(session: Session, payload: JobUpsert) -> Job:
    item = session.get(Job, payload.run_id)
    data = payload.model_dump(mode="json")
    if item is None:
        item = Job(**data)
        session.add(item)
    else:
        for key, value in data.items():
            setattr(item, key, value)
    item.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(item)
    return item


def rename_job(session: Session, run_id: str, name: str) -> Job:
    item = get_job(session, run_id)
    item.name = name
    item.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(item)
    return item


def delete_job(session: Session, run_id: str, *, force: bool = False) -> None:
    item = get_job(session, run_id)
    if not force and item.state in ACTIVE_JOB_STATES:
        raise ActiveJobError(f"Stop job before removing it: {run_id}")
    session.execute(delete(NodeLog).where(NodeLog.run_id == run_id))
    session.delete(item)
    session.commit()


def get_node_log(session: Session, run_id: str, node_id: str) -> NodeLog:
    item = session.execute(select(NodeLog).where(NodeLog.run_id == run_id, NodeLog.node_id == node_id)).scalar_one_or_none()
    if item is None:
        raise KeyError(f"Node log not found: {run_id}/{node_id}")
    return item


def upsert_node_log(session: Session, payload: NodeLogUpsert) -> NodeLog:
    item = session.execute(select(NodeLog).where(NodeLog.run_id == payload.run_id, NodeLog.node_id == payload.node_id)).scalar_one_or_none()
    data = payload.model_dump()
    if item is None:
        item = NodeLog(**data)
        session.add(item)
    else:
        for key, value in data.items():
            setattr(item, key, value)
    item.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(item)
    return item
