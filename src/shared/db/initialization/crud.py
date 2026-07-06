from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.common import many
from shared.db.initialization.models import Initialization
from shared.db.initialization.schemas import InitializationCreate


def list_initialization(session: Session) -> Sequence[Initialization]:
    return many(session, Initialization)


def create_initialization(session: Session, payload: InitializationCreate) -> Initialization:
    item = Initialization(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
