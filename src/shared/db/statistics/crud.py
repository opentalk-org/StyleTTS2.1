import uuid
from collections.abc import Sequence

from sqlalchemy import Integer, delete, select
from sqlalchemy.orm import Session

from shared.db.common import one
from shared.db.statistics.models import StatisticsEntry
from shared.db.statistics.schemas import StatisticsEntryCreate


def create_statistics_entry(session: Session, payload: StatisticsEntryCreate) -> StatisticsEntry:
    return bulk_create_statistics_entries(session, [payload])[0]


def bulk_create_statistics_entries(
    session: Session,
    payloads: Sequence[StatisticsEntryCreate],
) -> list[StatisticsEntry]:
    items = []
    for payload in payloads:
        data = payload.model_dump()
        data["metadata_"] = data.pop("metadata")
        items.append(StatisticsEntry(**data))
    if not items:
        return []
    session.add_all(items)
    session.commit()
    return items


def get_statistics_entry(session: Session, statistics_entry_id: uuid.UUID) -> StatisticsEntry:
    return one(session, StatisticsEntry, statistics_entry_id)


def list_statistics_entries(session: Session, dataset_id: uuid.UUID | None = None) -> Sequence[StatisticsEntry]:
    statement = select(StatisticsEntry).order_by(StatisticsEntry.created_at.desc())
    if dataset_id is not None:
        statement = statement.where(StatisticsEntry.dataset_id == dataset_id)
    return session.execute(statement).scalars().all()


def list_statistics_summaries(session: Session, dataset_id: uuid.UUID | None = None) -> list[dict]:
    file_count = StatisticsEntry.payload["file_count"].astext.cast(Integer)
    statement = select(
        StatisticsEntry.id,
        StatisticsEntry.name,
        StatisticsEntry.dataset_id,
        StatisticsEntry.created_at,
        file_count.label("file_count"),
    ).order_by(StatisticsEntry.created_at.desc())
    if dataset_id is not None:
        statement = statement.where(StatisticsEntry.dataset_id == dataset_id)
    return [
        {"id": row.id, "name": row.name, "dataset_id": row.dataset_id, "created_at": row.created_at, "file_count": row.file_count or 0}
        for row in session.execute(statement)
    ]


def delete_statistics_entry(session: Session, statistics_entry_id: uuid.UUID) -> None:
    result = session.execute(delete(StatisticsEntry).where(StatisticsEntry.id == statistics_entry_id))
    if result.rowcount == 0:
        raise KeyError(f"StatisticsEntry {statistics_entry_id} not found")
    session.commit()
