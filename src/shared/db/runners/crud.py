from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.common import many
from shared.db.runners.models import Runner
from shared.db.runners.schemas import RunnerCreate


def list_runners(session: Session) -> Sequence[Runner]:
    return many(session, Runner)


def create_runner(session: Session, payload: RunnerCreate) -> Runner:
    item = Runner(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def upsert_runner(session: Session, payload: RunnerCreate) -> Runner:
    statement = select(Runner).where(Runner.name == payload.name)
    item = session.execute(statement).scalar_one_or_none()
    if item is None:
        return create_runner(session, payload)
    item.hostname = payload.hostname
    item.port = payload.port
    item.gpu_index = payload.gpu_index
    item.resources = payload.resources
    session.commit()
    session.refresh(item)
    return item
