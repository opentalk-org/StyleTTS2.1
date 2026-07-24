from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import event
from sqlalchemy.orm import Session

from shared.storage import ObjectStore


@dataclass(frozen=True)
class StagedObject:
    store: ObjectStore
    path: str


STAGED_OBJECTS_KEY = "runflow_staged_objects"


def register_staged_object(
    session: Session,
    store: ObjectStore,
    path: str,
) -> None:
    staged = session.info.setdefault(STAGED_OBJECTS_KEY, [])
    staged.append(StagedObject(store, path))


@event.listens_for(Session, "after_commit")
def _accept_staged_objects(session: Session) -> None:
    session.info.pop(STAGED_OBJECTS_KEY, None)


@event.listens_for(Session, "after_rollback")
def _delete_staged_objects(session: Session) -> None:
    staged = session.info.pop(STAGED_OBJECTS_KEY, [])
    for item in staged:
        item.store.delete(item.path)
