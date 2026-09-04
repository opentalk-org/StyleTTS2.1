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
from shared.db.audio.clickhouse import get_audio_file as get_audio_record
from shared.db.audio.storage_locations import audio_storage_locations
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.db.waveforms.schemas import WaveformRead
from shared.storage import ObjectRange

router = APIRouter()
waveform_service = WaveformService()


@router.get("/{audio_file_id}/content")
async def audio_content(
    audio_file_id: uuid.UUID,
    range_header: str | None = Header(None, alias="Range"),
) -> Response:
    try:
        with database_session() as session:
            item = get_audio_record(audio_file_id)
            require_packed_audio(item)
            start, end = content_range(range_header, item.byte_length)
            location = audio_storage_locations(session, [audio_file_id])[audio_file_id]
            data = settings_crud.object_store(session).read_range(
                ObjectRange(
                    location.object_path,
                    location.byte_offset + start,
                    end - start + 1,
                )
            )
            return Response(
                data,
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                media_type=content_type(item.metadata),
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
            item = get_audio_record(audio_file_id)
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
        require_packed_audio(get_audio_record(audio_file_id))
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return WaveformStatusRead(status=await waveform_service.ensure(audio_file_id))
