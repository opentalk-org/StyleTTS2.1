import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.common import one
from shared.db.statistics.models import StatisticsEntry
from shared.db.statistics.schemas import StatisticsEntryCreate


def create_statistics_entry(session: Session, payload: StatisticsEntryCreate) -> StatisticsEntry:
    data = payload.model_dump()
    data["metadata_"] = data.pop("metadata")
    item = StatisticsEntry(**data)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def get_statistics_entry(session: Session, statistics_entry_id: uuid.UUID) -> StatisticsEntry:
    return one(session, StatisticsEntry, statistics_entry_id)


def list_statistics_entries(session: Session, dataset_id: uuid.UUID | None = None) -> Sequence[StatisticsEntry]:
    statement = select(StatisticsEntry).order_by(StatisticsEntry.created_at.desc())
    if dataset_id is not None:
        statement = statement.where(StatisticsEntry.dataset_id == dataset_id)
    return session.execute(statement).scalars().all()
