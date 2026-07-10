from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from backend.mos.schemas import MosAudioRead, MosPairRead, MosRatingRead
from shared.db import database_session
from shared.db.audio.models import AudioFile
from shared.db.mos import crud as mos_crud
from shared.db.mos.schemas import MosComparisonRead, MosRatingCreate


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
