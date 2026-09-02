from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from shared.db.datasets.clickhouse.crud import (
    add_audio_files,
    create_dataset as _create_dataset,
    delete_dataset as _delete_dataset,
    get_dataset,
    list_dataset_file_counts as _list_counts,
    list_datasets as _list_datasets,
    remove_audio_files,
)
from shared.db.datasets.clickhouse.models import DatasetRecord
from shared.db.datasets.clickhouse.training import (
    count_dataset_training_audio as _count_training,
    dataset_training_duration_totals as _duration_totals,
    dataset_training_minimum_duration as _minimum_duration,
    iter_dataset_training_audio as _iter_training,
    list_dataset_metadata_values as _metadata_values,
    list_tts_reference_candidates as _tts_reference_candidates,
)
from shared.db.datasets.schemas import DatasetCreate


def list_datasets() -> list[DatasetRecord]:
    return _list_datasets()


def list_dataset_file_counts():
    return _list_counts()


def get_dataset_by_name(name: str) -> DatasetRecord | None:
    return next((item for item in _list_datasets() if item.name == name), None)


def create_dataset(payload: DatasetCreate) -> DatasetRecord:
    return _create_dataset(
        DatasetRecord(id=uuid4(), updated_at=datetime.now(UTC), name=payload.name)
    )


def delete_dataset(dataset_id: UUID) -> None:
    _delete_dataset(dataset_id)


def bulk_add_audio_files_to_dataset(
    dataset_id: UUID,
    audio_file_ids: Sequence[UUID],
) -> None:
    now = datetime.now(UTC)
    add_audio_files(dataset_id, audio_file_ids, now, now)


def add_audio_file_to_dataset(dataset_id: UUID, audio_file_id: UUID) -> DatasetRecord:
    bulk_add_audio_files_to_dataset(dataset_id, [audio_file_id])
    return get_dataset(dataset_id)


def bulk_remove_audio_files_from_dataset(
    dataset_id: UUID,
    audio_file_ids: Sequence[UUID],
) -> None:
    get_dataset(dataset_id)
    remove_audio_files(dataset_id, audio_file_ids)


def remove_audio_file_from_dataset(dataset_id: UUID, audio_file_id: UUID) -> DatasetRecord:
    bulk_remove_audio_files_from_dataset(dataset_id, [audio_file_id])
    return get_dataset(dataset_id)


def count_dataset_training_audio(dataset_id: UUID) -> int:
    return _count_training(dataset_id)


def iter_dataset_training_audio(
    dataset_id: UUID,
    descending: bool = False,
    duration_above: float | None = None,
    duration_at_most: float | None = None,
    excluded_audio_ids: set[UUID] | None = None,
    audio_id_after: UUID | None = None,
    audio_id_at_most: UUID | None = None,
):
    return _iter_training(
        dataset_id,
        descending,
        duration_above,
        duration_at_most,
        excluded_audio_ids,
        audio_id_after,
        audio_id_at_most,
    )


def dataset_training_duration_totals(
    dataset_id: UUID,
    upper_bounds: tuple[float, ...],
    excluded_audio_ids: set[UUID],
):
    return _duration_totals(dataset_id, upper_bounds, excluded_audio_ids)


def dataset_training_minimum_duration(
    dataset_id: UUID,
    max_duration: float,
    excluded_audio_ids: set[UUID],
) -> float:
    return _minimum_duration(dataset_id, max_duration, excluded_audio_ids)


def list_dataset_metadata_values(dataset_id: UUID, key: str) -> set[str]:
    get_dataset(dataset_id)
    return _metadata_values(dataset_id, key)


def list_tts_reference_candidates(dataset_ids: Sequence[UUID], streams: Sequence[str]):
    return _tts_reference_candidates(dataset_ids, streams)
