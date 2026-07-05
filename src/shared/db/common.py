import uuid
from collections.abc import Sequence
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.base import Base


ModelT = TypeVar("ModelT", bound=Base)


def one(session: Session, model: type[ModelT], item_id: uuid.UUID) -> ModelT:
    result = session.execute(select(model).where(model.id == item_id)).scalar_one_or_none()
    if result is None:
        raise KeyError(f"{model.__name__} not found: {item_id}")
    return result


def many(session: Session, model: type[ModelT]) -> Sequence[ModelT]:
    return session.execute(select(model)).scalars().all()
