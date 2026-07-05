import uuid
from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.common import many, one
from shared.db.datasets.models import Dataset
from shared.db.datasets.schemas import DatasetCreate


def list_datasets(session: Session) -> Sequence[Dataset]:
    return many(session, Dataset)


def create_dataset(session: Session, payload: DatasetCreate) -> Dataset:
    item = Dataset(**payload.model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


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
