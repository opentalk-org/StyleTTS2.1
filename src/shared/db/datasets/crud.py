import uuid
from collections.abc import Sequence

from sqlalchemy import case, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.common import many, one
from shared.db.datasets.models import Dataset, dataset_audio_files
from shared.db.datasets.schemas import (
    DatasetCreate,
    DatasetDurationBins,
    DatasetTrainingAudio,
)


def list_datasets(session: Session) -> Sequence[Dataset]:
    return many(session, Dataset)


def iter_dataset_training_audio(
    session: Session,
    dataset_id: uuid.UUID,
    descending: bool = False,
    duration_above: float | None = None,
    duration_at_most: float | None = None,
    excluded_audio_ids: set[uuid.UUID] | None = None,
):
    order = AudioFile.id.desc() if descending else AudioFile.id
    statement = (
        select(
            AudioFile.id,
            AudioFile.duration,
            AudioFile.speaker_id,
            AudioFile.language,
            AudioFile.segments,
            AudioFile.metadata_,
        )
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.virtual.is_(False),
        )
        .order_by(order)
        .execution_options(yield_per=2_000)
    )
    if duration_above is not None:
        statement = statement.where(AudioFile.duration > duration_above)
    if duration_at_most is not None:
        statement = statement.where(AudioFile.duration <= duration_at_most)
    if excluded_audio_ids:
        statement = statement.where(AudioFile.id.not_in(excluded_audio_ids))
    for row in session.execute(statement):
        yield DatasetTrainingAudio(*row)


def count_dataset_training_audio(
    session: Session,
    dataset_id: uuid.UUID,
) -> int:
    statement = (
        select(func.count())
        .select_from(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.virtual.is_(False),
        )
    )
    return int(session.scalar(statement))


def dataset_training_duration_totals(
    session: Session,
    dataset_id: uuid.UUID,
    upper_bounds: tuple[float, ...],
    excluded_audio_ids: set[uuid.UUID],
) -> DatasetDurationBins:
    bin_index = case(
        *((AudioFile.duration <= boundary, index)
          for index, boundary in enumerate(upper_bounds)),
    )
    statement = (
        select(bin_index, func.sum(AudioFile.duration), func.count())
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.virtual.is_(False),
            AudioFile.duration > 0,
            AudioFile.duration <= upper_bounds[-1],
        )
        .group_by(bin_index)
    )
    if excluded_audio_ids:
        statement = statement.where(AudioFile.id.not_in(excluded_audio_ids))
    rows = session.execute(statement).all()
    totals_by_bin = {int(index): float(total) for index, total, _ in rows}
    return DatasetDurationBins(
        total_seconds=tuple(
            totals_by_bin.get(index, 0.0)
            for index in range(len(upper_bounds))
        ),
        file_count=sum(int(count) for _, _, count in rows),
    )


def dataset_training_minimum_duration(
    session: Session,
    dataset_id: uuid.UUID,
    max_duration: float,
    excluded_audio_ids: set[uuid.UUID],
) -> float:
    statement = (
        select(func.min(AudioFile.duration))
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.virtual.is_(False),
            AudioFile.duration > 0,
            AudioFile.duration <= max_duration,
        )
    )
    if excluded_audio_ids:
        statement = statement.where(AudioFile.id.not_in(excluded_audio_ids))
    minimum = session.scalar(statement)
    if minimum is None:
        raise ValueError("training dataset has no audio within the duration limit")
    return float(minimum)


def get_dataset_by_name(session: Session, name: str) -> Dataset | None:
    return session.execute(select(Dataset).where(Dataset.name == name)).scalar_one_or_none()


def list_dataset_audio_files_by_stage1_slug(
    session: Session,
    dataset_id: uuid.UUID,
    slug: str,
) -> Sequence[AudioFile]:
    statement = (
        select(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            AudioFile.metadata_["stage1_dataset"].astext == slug,
        )
        .order_by(AudioFile.id)
    )
    return session.execute(statement).scalars().all()


def list_dataset_file_counts(session: Session) -> Sequence[tuple[Dataset, int]]:
    statement = (
        select(Dataset, func.count(dataset_audio_files.c.audio_file_id))
        .outerjoin(dataset_audio_files, Dataset.id == dataset_audio_files.c.dataset_id)
        .group_by(Dataset.id)
        .order_by(Dataset.name)
    )
    return [(dataset, file_count) for dataset, file_count in session.execute(statement).all()]


def list_dataset_metadata_values(
    session: Session,
    dataset_id: uuid.UUID,
    key: str,
) -> set[str]:
    one(session, Dataset, dataset_id)
    value = AudioFile.metadata_[key].astext
    statement = (
        select(value)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
            value.is_not(None),
        )
    )
    return {str(item) for item in session.scalars(statement)}


def list_tts_reference_candidates(
    session: Session,
    dataset_ids: Sequence[uuid.UUID],
    streams: Sequence[str],
) -> Sequence[AudioFile]:
    for dataset_id in dataset_ids:
        one(session, Dataset, dataset_id)
    statement = (
        select(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id.in_(dataset_ids),
            AudioFile.metadata_["stream"].astext.in_(streams),
        )
        .order_by(
            AudioFile.metadata_["stream"].astext,
            AudioFile.id,
        )
    )
    return session.execute(statement).unique().scalars().all()


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


def bulk_add_audio_files_to_dataset(
    session: Session,
    dataset_id: uuid.UUID,
    audio_file_ids: Sequence[uuid.UUID],
    commit: bool = True,
) -> None:
    one(session, Dataset, dataset_id)
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    statement = (
        insert(dataset_audio_files)
        .values([{"dataset_id": dataset_id, "audio_file_id": audio_file_id} for audio_file_id in ids])
        .on_conflict_do_nothing(index_elements=["dataset_id", "audio_file_id"])
    )
    session.execute(statement)
    if commit:
        session.commit()


def remove_audio_file_from_dataset(session: Session, dataset_id: uuid.UUID, audio_file_id: uuid.UUID) -> Dataset:
    bulk_remove_audio_files_from_dataset(session, dataset_id, [audio_file_id])
    dataset = one(session, Dataset, dataset_id)
    session.refresh(dataset)
    return dataset


def bulk_remove_audio_files_from_dataset(
    session: Session,
    dataset_id: uuid.UUID,
    audio_file_ids: Sequence[uuid.UUID],
    commit: bool = True,
) -> None:
    one(session, Dataset, dataset_id)
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    statement = delete(dataset_audio_files).where(
        dataset_audio_files.c.dataset_id == dataset_id,
        dataset_audio_files.c.audio_file_id.in_(ids),
    )
    session.execute(statement)
    if commit:
        session.commit()
