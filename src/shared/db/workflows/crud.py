from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.common import many
from shared.db.workflows.models import Workflow
from shared.db.workflows.schemas import WorkflowCreate


def list_workflows(session: Session) -> Sequence[Workflow]:
    return many(session, Workflow)


def create_workflow(session: Session, payload: WorkflowCreate) -> Workflow:
    item = Workflow(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
