from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from shared.db.audio import crud as audio_crud
from shared.db.mos.clickhouse.crud import (
    count_comparisons as _count_comparisons,
    create_comparison,
    delete_comparison,
    get_comparison,
    list_comparisons as _list_comparisons,
    sample_pair as _sample_pair,
    update_comparison,
)
from shared.db.mos.clickhouse.models import MosComparisonRecord, MosPairIds
from shared.db.mos.schemas import MosRatingCreate, MosRatingUpdate


def sample_pair(dataset_ids: Sequence[UUID]) -> MosPairIds:
    if not dataset_ids:
        raise ValueError("MOS pair requires at least one dataset")
    errors = []
    for dataset_id in dataset_ids:
        try:
            return _sample_pair(dataset_id)
        except ValueError as error:
            errors.append(error)
    raise ValueError(
        "selected datasets do not contain two eligible audio files"
    ) from errors[-1]


def create_rating(payload: MosRatingCreate) -> MosComparisonRecord:
    audio_files = audio_crud.get_audio_files_bulk(
        [payload.audio_a_id, payload.audio_b_id]
    )
    if len(audio_files) != 2:
        raise ValueError("both MOS audio files must exist")
    now = datetime.now(UTC)
    return create_comparison(
        MosComparisonRecord(
            id=uuid4(),
            updated_at=now,
            audio_a_id=payload.audio_a_id,
            audio_b_id=payload.audio_b_id,
            preferred_audio_id=payload.preferred_audio_id,
            score_a=payload.score_a,
            score_b=payload.score_b,
            created_at=now,
        )
    )


def list_comparisons(dataset_id: UUID) -> list[MosComparisonRecord]:
    return list(reversed(_list_comparisons(dataset_id, 4_294_967_295)))


def count_comparisons(dataset_id: UUID) -> int:
    return _count_comparisons(dataset_id)


def iter_comparisons(dataset_id: UUID) -> Iterator[MosComparisonRecord]:
    yield from list_comparisons(dataset_id)


def list_comparisons_page(
    dataset_ids: Sequence[UUID], limit: int, offset: int
) -> tuple[list[MosComparisonRecord], int]:
    if len(dataset_ids) != 1:
        raise ValueError("exactly one dataset is required")
    return _list_comparisons(dataset_ids[0], limit, offset), _count_comparisons(
        dataset_ids[0]
    )


def comparison_audio_files(comparisons: Sequence[MosComparisonRecord]):
    ids = [
        audio_id
        for item in comparisons
        for audio_id in (item.audio_a_id, item.audio_b_id)
    ]
    return audio_crud.get_audio_files_bulk(ids)


def update_latest_rating(
    comparison_id: UUID, payload: MosRatingUpdate
) -> MosComparisonRecord:
    current = get_comparison(comparison_id)
    return update_comparison(
        current.model_copy(
            update={**payload.model_dump(), "updated_at": datetime.now(UTC)}
        )
    )


def undo_latest_rating(comparison_id: UUID) -> None:
    get_comparison(comparison_id)
    delete_comparison(comparison_id)
