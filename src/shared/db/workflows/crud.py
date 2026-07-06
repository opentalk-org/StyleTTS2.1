from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.common import many
from shared.db.workflows.models import Workflow
from shared.db.workflows.schemas import WorkflowCreate


def list_workflows(session: Session) -> Sequence[Workflow]:
    return many(session, Workflow)


def get_workflow(session: Session, workflow_id) -> Workflow:
    item = session.get(Workflow, workflow_id)
    if item is None:
        raise KeyError(f"Workflow not found: {workflow_id}")
    return item


def create_workflow(session: Session, payload: WorkflowCreate) -> Workflow:
    data = payload.model_dump(mode="json")
    item = Workflow(**data)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
