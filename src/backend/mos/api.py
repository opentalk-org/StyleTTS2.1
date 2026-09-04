from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from backend.mos.schemas import (
    MosAudioRead,
    MosPairRead,
    MosRatingDetailRead,
    MosRatingPage,
    MosRatingRead,
)
from shared.audio_annotations import AudioAnnotations
from shared.db.audio.clickhouse import AudioFileRecord, get_audio_file
from shared.db.mos import crud as mos_crud
from shared.db.mos.clickhouse import MosComparisonRecord
from shared.db.mos.schemas import MosComparisonRead, MosRatingCreate, MosRatingUpdate


router = APIRouter(prefix="/mos", tags=["mos"])


@router.get("/pair", response_model=MosPairRead)
async def get_mos_pair(dataset_id: list[UUID] = Query()) -> MosPairRead:
    try:
        if not dataset_id:
            raise ValueError("at least one dataset is required")
        pair = mos_crud.sample_pair(dataset_id)
        return MosPairRead(
            dataset_id=pair.dataset_id,
            audio_a=_audio_response(get_audio_file(pair.audio_a_id)),
            audio_b=_audio_response(get_audio_file(pair.audio_b_id)),
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


@router.post(
    "/ratings", response_model=MosRatingRead, status_code=status.HTTP_201_CREATED
)
async def create_mos_rating(payload: MosRatingCreate) -> MosRatingRead:
    try:
        comparison = mos_crud.create_rating(payload)
        return MosRatingRead.model_validate(comparison.model_dump())
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


@router.get("/ratings", response_model=MosRatingPage)
async def list_mos_ratings(
    dataset_id: list[UUID] = Query(),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> MosRatingPage:
    try:
        if len(dataset_id) != 1:
            raise ValueError("exactly one dataset is required")
        rows, total = mos_crud.list_comparisons_page(dataset_id, limit, offset)
        audio_files = _comparison_audio_files(rows)
        return MosRatingPage(
            rows=[_rating_response(row, audio_files, True) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


@router.patch("/ratings/{comparison_id}", response_model=MosRatingDetailRead)
async def update_mos_rating(
    comparison_id: UUID, payload: MosRatingUpdate
) -> MosRatingDetailRead:
    try:
        comparison = mos_crud.update_latest_rating(comparison_id, payload)
        return _rating_response(comparison, _comparison_audio_files([comparison]), True)
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


@router.delete("/ratings/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mos_rating(comparison_id: UUID) -> None:
    try:
        mos_crud.undo_latest_rating(comparison_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


def _audio_response(item: AudioFileRecord) -> MosAudioRead:
    return MosAudioRead(
        id=item.id,
        name=item.name,
        duration=item.duration,
        annotations=AudioAnnotations(
            score=item.score,
            accuracy=None,
            metadata=item.metadata,
        ),
    )


def _rating_response(
    comparison: MosComparisonRecord,
    audio_files: dict[UUID, AudioFileRecord],
    can_modify: bool,
) -> MosRatingDetailRead:
    summary = MosComparisonRead.model_validate(comparison)
    return MosRatingDetailRead(
        **summary.model_dump(),
        audio_a=_audio_response(audio_files[comparison.audio_a_id]),
        audio_b=_audio_response(audio_files[comparison.audio_b_id]),
        can_modify=can_modify,
    )


def _comparison_audio_files(
    comparisons: list[MosComparisonRecord],
) -> dict[UUID, AudioFileRecord]:
    return mos_crud.comparison_audio_files(comparisons)
