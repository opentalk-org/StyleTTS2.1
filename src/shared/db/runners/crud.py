from collections.abc import Sequence

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
