import json
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Response, UploadFile, status

from backend.audio.schemas import AudioFileListItem, AudioFilePage, AudioSegmentRead, AudioSegmentWrite, AudioSort
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioCreate, AudioPartRead
from shared.db.audio.schemas import AudioUpdate
from shared.db.datasets import crud as dataset_crud
from shared.db.waveforms import crud as waveform_crud
from shared.db.waveforms.schemas import WaveformInput, WaveformRead


router = APIRouter(prefix="/audio-files", tags=["audio-files"])
DEFAULT_STREAM_CHUNK = 1024 * 1024


@router.get("", response_model=AudioFilePage)
async def list_audio_files(
    query: str = "",
    dataset: str = "all",
    sort: AudioSort = "updated",
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AudioFilePage:
    with database_session() as session:
        rows, total = audio_crud.search_audio_files(session, query, dataset, sort, limit, offset)
        return AudioFilePage(rows=[audio_response(item, 8) for item in rows], total=total)


@router.post("/upload", response_model=AudioFileListItem, status_code=status.HTTP_201_CREATED)
async def upload_audio_file(
    file: UploadFile = File(),
    duration: float = Form(),
    sample_rate: int = Form(),
    dataset_id: str = Form(""),
    speaker: str = Form(""),
    waveform: str = Form(),
) -> AudioFileListItem:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is empty")
    if duration <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio duration must be positive")
    if sample_rate <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio sample rate must be positive")
    if file.filename is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio filename is required")
    try:
        with database_session() as session:
            item = audio_crud.create_audio_file(session, _audio_payload(file, data, duration, sample_rate, speaker, waveform))
            if dataset_id:
                dataset_crud.add_audio_file_to_dataset(session, uuid.UUID(dataset_id), item.id)
                session.refresh(item, attribute_names=["datasets"])
            return audio_response(item, None)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_matching_audio_files(
    query: str = Query(min_length=0),
    dataset: str = Query(min_length=1),
) -> None:
    try:
        with database_session() as session:
            ids = audio_crud.search_audio_file_ids(session, query, dataset)
            audio_crud.bulk_delete_audio_files(session, ids)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("/{audio_file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_audio_file(audio_file_id: uuid.UUID) -> None:
    try:
        with database_session() as session:
            audio_crud.delete_audio_file(session, audio_file_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/{audio_file_id}", response_model=AudioFileListItem)
async def get_audio_file(audio_file_id: uuid.UUID) -> AudioFileListItem:
    try:
        with database_session() as session:
            return audio_response(audio_crud.get_audio_file(session, audio_file_id), None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/{audio_file_id}/content")
async def audio_content(audio_file_id: uuid.UUID, range_header: str | None = Header(None, alias="Range")) -> Response:
    try:
        with database_session() as session:
            item = audio_crud.get_audio_file(session, audio_file_id)
            start, end = _content_range(range_header, item.byte_length)
            data = audio_crud.read_audio_part(session, audio_file_id, AudioPartRead(start=start, length=end - start + 1))
            return Response(
                data,
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                media_type=_content_type(item.metadata_),
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {start}-{end}/{item.byte_length}",
                },
            )
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


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
            return waveform_crud.read_waveform(session, audio_file_id, start, end or item.duration, points)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.put("/{audio_file_id}/segments", response_model=AudioFileListItem)
async def replace_audio_segments(audio_file_id: uuid.UUID, payload: list[AudioSegmentWrite]) -> AudioFileListItem:
    try:
        with database_session() as session:
            item = audio_crud.get_audio_file(session, audio_file_id)
            updated = audio_crud.update_audio_file(
                session,
                audio_file_id,
                AudioUpdate(
                    name=item.name,
                    wav_bytes=None,
                    duration=item.duration,
                    segments=[segment.model_dump(mode="json") for segment in payload],
                    metadata=item.metadata_,
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


def audio_response(item: AudioFile, segment_limit: int | None) -> AudioFileListItem:
    metadata = dict(item.metadata_)
    segments = item.segments if segment_limit is None else item.segments[:segment_limit]
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        speaker=_speaker(metadata),
        duration=item.duration,
        sample_rate=_sample_rate(metadata),
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=len(item.segments),
        segment_preview=[segment_response(segment) for segment in segments],
        dataset_ids=[dataset.id for dataset in item.datasets],
        virtual=item.virtual,
        metadata=metadata,
        updated_at=item.updated_at,
    )


def _audio_payload(file: UploadFile, data: bytes, duration: float, sample_rate: int, speaker: str, waveform: str) -> AudioCreate:
    metadata: dict[str, Any] = {"sample_rate": sample_rate}
    if file.content_type is not None:
        metadata["content_type"] = file.content_type
    metadata["source_filename"] = file.filename
    if speaker:
        metadata["speaker"] = speaker
    return AudioCreate(
        name=file.filename,
        wav_bytes=data,
        duration=duration,
        segments=[],
        metadata=metadata,
        virtual=False,
        waveform=_waveform_payload(waveform),
    )


def segment_response(segment: dict[str, Any]) -> AudioSegmentRead:
    return AudioSegmentRead(
        id=str(segment["id"]),
        start=float(segment["start"]),
        end=float(segment["end"]),
        text=str(segment["text"]) if "text" in segment else "",
        phon=str(segment["phon"]) if "phon" in segment else "",
        speaker=_segment_speaker(segment),
    )


def _speaker(metadata: dict[str, Any]) -> str:
    if "speaker" in metadata:
        return str(metadata["speaker"])
    if "voice" in metadata:
        return str(metadata["voice"])
    if "voice_id" in metadata:
        return str(metadata["voice_id"])
    return "-"


def _sample_rate(metadata: dict[str, Any]) -> int | None:
    if "sample_rate" not in metadata:
        return None
    return int(metadata["sample_rate"])


def _content_type(metadata: dict[str, Any]) -> str:
    if "content_type" in metadata:
        return str(metadata["content_type"])
    return "application/octet-stream"


def _waveform_payload(raw: str) -> WaveformInput:
    return WaveformInput.model_validate(json.loads(raw))


def _content_range(range_header: str | None, byte_length: int) -> tuple[int, int]:
    if range_header is None:
        return 0, min(byte_length, DEFAULT_STREAM_CHUNK) - 1
    unit, value = range_header.split("=", 1)
    if unit != "bytes":
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Only byte ranges are supported")
    start_text, end_text = value.split("-", 1)
    start = int(start_text) if start_text else 0
    end = int(end_text) if end_text else min(byte_length - 1, start + DEFAULT_STREAM_CHUNK - 1)
    if start < 0 or end < start or start >= byte_length:
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Requested range is invalid")
    return start, min(end, byte_length - 1)


def _segment_speaker(segment: dict[str, Any]) -> str:
    if "speaker" in segment:
        return str(segment["speaker"])
    if "voice" in segment:
        return str(segment["voice"])
    if "voice_id" in segment and segment["voice_id"] is not None:
        return str(uuid.UUID(str(segment["voice_id"])))
    return "-"
