from fastapi import APIRouter, HTTPException, Query, status

from shared.db import database_session
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import SpeakerPage, SpeakerRename


router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.get("", response_model=SpeakerPage)
async def list_speakers(
    query: str = "",
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> SpeakerPage:
    with database_session() as session:
        rows, total = speaker_crud.search_speakers(session, query, limit, offset)
        return SpeakerPage(rows=rows, total=total)


@router.patch("/{speaker_id:path}", status_code=status.HTTP_204_NO_CONTENT)
async def rename_speaker(speaker_id: str, request: SpeakerRename) -> None:
    try:
        with database_session() as session:
            speaker_crud.rename_speaker(session, speaker_id, request.speaker_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_matching_speakers(query: str = Query("", min_length=0)) -> None:
    with database_session() as session:
        speaker_crud.clear_matching_speakers(session, query)


@router.delete("/{speaker_id:path}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_speaker(speaker_id: str) -> None:
    try:
        with database_session() as session:
            speaker_crud.clear_speaker(session, speaker_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
