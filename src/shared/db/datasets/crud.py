import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.common import many, one
from shared.db.datasets.models import Dataset, dataset_audio_files
from shared.db.datasets.schemas import DatasetCreate


def list_datasets(session: Session) -> Sequence[Dataset]:
    return many(session, Dataset)


def list_dataset_file_counts(session: Session) -> Sequence[tuple[Dataset, int]]:
    statement = (
        select(Dataset, func.count(dataset_audio_files.c.audio_file_id))
        .outerjoin(dataset_audio_files, Dataset.id == dataset_audio_files.c.dataset_id)
        .group_by(Dataset.id)
        .order_by(Dataset.name)
    )
    return [(dataset, file_count) for dataset, file_count in session.execute(statement).all()]


def create_dataset(session: Session, payload: DatasetCreate) -> Dataset:
    item = Dataset(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def delete_dataset(session: Session, dataset_id: uuid.UUID) -> None:
    session.execute(delete(dataset_audio_files).where(dataset_audio_files.c.dataset_id == dataset_id))
    result = session.execute(delete(Dataset).where(Dataset.id == dataset_id))
    if result.rowcount != 1:
        raise KeyError(f"Dataset not found: {dataset_id}")
    session.commit()


def add_audio_file_to_dataset(session: Session, dataset_id: uuid.UUID, audio_file_id: uuid.UUID) -> Dataset:
    dataset = one(session, Dataset, dataset_id)
    audio_file = one(session, AudioFile, audio_file_id)
    dataset.audio_files.append(audio_file)
    session.commit()
    session.refresh(dataset)
    return dataset


def remove_audio_file_from_dataset(session: Session, dataset_id: uuid.UUID, audio_file_id: uuid.UUID) -> Dataset:
    dataset = one(session, Dataset, dataset_id)
    audio_file = one(session, AudioFile, audio_file_id)
    dataset.audio_files.remove(audio_file)
    session.commit()
    session.refresh(dataset)
    return dataset
