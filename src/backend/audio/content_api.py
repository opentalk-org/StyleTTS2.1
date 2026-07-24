import uuid

from fastapi import APIRouter, Header, HTTPException, Query, Response, status

from backend.audio.responses import (
    content_range,
    content_type,
    require_packed_audio,
)
from backend.audio.schemas import WaveformStatusRead
from backend.audio.waveform_service import WaveformService
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioPartRead
from shared.db.waveforms import crud as waveform_crud
from shared.db.waveforms.schemas import WaveformRead

router = APIRouter()
waveform_service = WaveformService()


@router.get("/{audio_file_id}/content")
async def audio_content(
    audio_file_id: uuid.UUID,
    range_header: str | None = Header(None, alias="Range"),
) -> Response:
    try:
        with database_session() as session:
            item = audio_crud.get_audio_file(session, audio_file_id)
            require_packed_audio(item)
            start, end = content_range(range_header, item.byte_length)
            data = audio_crud.read_audio_part(
                session,
                audio_file_id,
                AudioPartRead(start=start, length=end - start + 1),
            )
            return Response(
                data,
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                media_type=content_type(item.metadata_),
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {start}-{end}/{item.byte_length}",
                },
            )
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.get("/{audio_file_id}/waveform", response_model=WaveformRead)
async def get_waveform(
    audio_file_id: uuid.UUID,
    start: float = Query(0, ge=0),
    end: float | None = Query(None, gt=0),
    points: int = Query(1200, ge=1, le=10000),
) -> WaveformRead:
    try:
        with database_session() as session:
            item = audio_crud.get_audio_file(session, audio_file_id)
            require_packed_audio(item)
            return waveform_crud.read_waveform(
                session,
                audio_file_id,
                start,
                end if end is not None else item.duration,
                points,
            )
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post("/{audio_file_id}/waveform", response_model=WaveformStatusRead)
async def ensure_waveform(audio_file_id: uuid.UUID) -> WaveformStatusRead:
    try:
        with database_session() as session:
            require_packed_audio(audio_crud.get_audio_file(session, audio_file_id))
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return WaveformStatusRead(status=await waveform_service.ensure(audio_file_id))
