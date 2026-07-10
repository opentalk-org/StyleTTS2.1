import uuid

from fastapi import APIRouter, HTTPException, Query, status

from shared.db import database_session
from shared.db.voices import crud as voice_crud
from shared.db.voices.models import Voice
from shared.db.voices.schemas import VoiceCreate, VoicePage, VoiceRead


router = APIRouter(prefix="/voices")


def voice_response(item: Voice) -> VoiceRead:
    return VoiceRead(id=item.id, name=item.name, segments=0, datasets=[])


@router.get("", response_model=VoicePage)
async def list_voices(
    query: str = "",
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> VoicePage:
    with database_session() as session:
        rows, total = voice_crud.search_voices(session, query, limit, offset)
        return VoicePage(rows=[voice_response(item) for item in rows], total=total)


@router.post("", response_model=VoiceRead, status_code=status.HTTP_201_CREATED)
async def create_voice(request: VoiceCreate) -> VoiceRead:
    try:
        with database_session() as session:
            item = voice_crud.create_voice(session, request)
            return voice_response(item)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.patch("/{voice_id}", response_model=VoiceRead)
async def rename_voice(voice_id: str, request: VoiceCreate) -> VoiceRead:
    try:
        with database_session() as session:
            item = voice_crud.rename_voice(session, uuid.UUID(voice_id), request.name)
            return voice_response(item)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_matching_voices(query: str = Query("", min_length=0)) -> None:
    with database_session() as session:
        ids = voice_crud.search_voice_ids(session, query)
        voice_crud.bulk_delete_voices(session, ids)


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(voice_id: str) -> None:
    try:
        with database_session() as session:
            voice_crud.delete_voice(session, uuid.UUID(voice_id))
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
