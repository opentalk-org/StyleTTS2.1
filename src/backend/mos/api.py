from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from backend.mos.schemas import MosAudioRead, MosPairRead, MosRatingDetailRead, MosRatingPage, MosRatingRead
from shared.db import database_session
from shared.db.audio.models import AudioFile
from shared.db.mos import crud as mos_crud
from shared.db.mos.models import MosComparison
from shared.db.mos.schemas import MosComparisonRead, MosRatingCreate, MosRatingUpdate


router = APIRouter(prefix="/mos", tags=["mos"])


@router.get("/pair", response_model=MosPairRead)
async def get_mos_pair(dataset_id: list[UUID] = Query()) -> MosPairRead:
    try:
        with database_session() as session:
            pair = mos_crud.sample_pair(session, dataset_id)
            return MosPairRead(
                dataset_id=pair.dataset_id,
                audio_a=_audio_response(pair.audio_a),
                audio_b=_audio_response(pair.audio_b),
            )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.post("/ratings", response_model=MosRatingRead, status_code=status.HTTP_201_CREATED)
async def create_mos_rating(payload: MosRatingCreate) -> MosRatingRead:
    try:
        with database_session() as session:
            comparison = mos_crud.create_rating(session, payload)
            return MosRatingRead.model_validate(
                MosComparisonRead.model_validate(comparison).model_dump()
            )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/ratings", response_model=MosRatingPage)
async def list_mos_ratings(
    dataset_id: list[UUID] = Query(),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> MosRatingPage:
    try:
        with database_session() as session:
            rows, total = mos_crud.list_comparisons_page(session, dataset_id, limit, offset)
            audio_files = mos_crud.comparison_audio_files(session, rows)
            return MosRatingPage(
                rows=[_rating_response(row, audio_files, True) for row in rows],
                total=total,
                limit=limit,
                offset=offset,
            )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.patch("/ratings/{comparison_id}", response_model=MosRatingDetailRead)
async def update_mos_rating(comparison_id: UUID, payload: MosRatingUpdate) -> MosRatingDetailRead:
    try:
        with database_session() as session:
            comparison = mos_crud.update_latest_rating(session, comparison_id, payload)
            audio_files = mos_crud.comparison_audio_files(session, [comparison])
            return _rating_response(comparison, audio_files, True)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("/ratings/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mos_rating(comparison_id: UUID) -> None:
    try:
        with database_session() as session:
            mos_crud.undo_latest_rating(session, comparison_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


def _audio_response(item: AudioFile) -> MosAudioRead:
    metadata = item.metadata_
    if "speaker" in metadata:
        speaker = str(metadata["speaker"])
    elif "voice" in metadata:
        speaker = str(metadata["voice"])
    else:
        speaker = ""
    return MosAudioRead(
        id=item.id,
        name=item.name,
        duration=item.duration,
        score=item.score,
        speaker=speaker,
    )


def _rating_response(
    comparison: MosComparison,
    audio_files: dict[UUID, AudioFile],
    can_modify: bool,
) -> MosRatingDetailRead:
    summary = MosComparisonRead.model_validate(comparison)
    return MosRatingDetailRead(
        **summary.model_dump(),
        audio_a=_audio_response(audio_files[comparison.audio_a_id]),
        audio_b=_audio_response(audio_files[comparison.audio_b_id]),
        can_modify=can_modify,
    )
