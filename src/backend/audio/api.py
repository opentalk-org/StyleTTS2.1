import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.audio.content_api import router as content_router
from backend.audio.responses import (
    audio_annotations,
    audio_list_response,
    audio_payload,
    audio_response,
)
from backend.audio.schemas import AddToDatasetRequest, AudioFileListItem, AudioFilePage, AudioLanguagePayload, AudioRenamePayload, AudioScorePayload, AudioSegmentWrite, AudioSort, AudioStylePromptPayload, AudioVoicePromptPayload
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioCreate, AudioUpdate
from shared.db.datasets import crud as dataset_crud


router = APIRouter(prefix="/audio-files", tags=["audio-files"])


@router.get("", response_model=AudioFilePage)
async def list_audio_files(
    query: str = "",
    language: str = "",
    dataset: str = "all",
    sort: AudioSort = "updated",
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AudioFilePage:
    with database_session() as session:
        rows, total = audio_crud.search_audio_files(
            session,
            query,
            dataset,
            sort,
            limit,
            offset,
            preview_limit=8,
            language=language,
        )
        return AudioFilePage(
            rows=[audio_list_response(item, segment_count, segment_preview) for item, segment_count, segment_preview in rows],
            total=total,
        )


@router.post("/upload", response_model=AudioFileListItem, status_code=status.HTTP_201_CREATED)
async def upload_audio_file(
    file: UploadFile = File(),
    duration: float = Form(),
    sample_rate: int = Form(),
    dataset_id: str = Form(""),
    speaker_id: str = Form(""),
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
        payload = audio_payload(file, data, duration, sample_rate, speaker_id)
        return await run_in_threadpool(_persist_uploaded_audio, payload, dataset_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


def _persist_uploaded_audio(payload: AudioCreate, dataset_id: str) -> AudioFileListItem:
    with database_session() as session:
        item = audio_crud.create_audio_file(session, payload)
        if dataset_id:
            dataset_crud.add_audio_file_to_dataset(session, uuid.UUID(dataset_id), item.id)
            session.refresh(item, attribute_names=["datasets"])
        return audio_response(item, None)


@router.post("/dataset-membership", status_code=status.HTTP_204_NO_CONTENT)
async def add_audio_files_to_dataset(payload: AddToDatasetRequest) -> None:
    try:
        dataset_id = uuid.UUID(payload.dataset_id)
        with database_session() as session:
            if payload.mode == "filter":
                ids = audio_crud.search_audio_file_ids(
                    session,
                    payload.query,
                    payload.dataset,
                    payload.language,
                )
            else:
                ids = [uuid.UUID(file_id) for file_id in payload.audio_file_ids]
            dataset_crud.bulk_add_audio_files_to_dataset(session, dataset_id, ids)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_matching_audio_files(
    query: str = Query(min_length=0),
    language: str = Query(default="", min_length=0),
    dataset: str = Query(min_length=1),
) -> None:
    try:
        with database_session() as session:
            ids = audio_crud.search_audio_file_ids(session, query, dataset, language)
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


@router.get("/by-run/{run_id}", response_model=list[AudioFileListItem])
async def list_audio_files_by_run(run_id: str) -> list[AudioFileListItem]:
    with database_session() as session:
        return [audio_response(item, 0) for item in audio_crud.list_audio_files_by_run(session, run_id)]


@router.get("/{audio_file_id}", response_model=AudioFileListItem)
async def get_audio_file(audio_file_id: uuid.UUID) -> AudioFileListItem:
    try:
        with database_session() as session:
            return audio_response(audio_crud.get_audio_file(session, audio_file_id), None)
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
                    annotations=audio_annotations(item),
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.patch("/{audio_file_id}/name", response_model=AudioFileListItem)
async def rename_audio_file(audio_file_id: uuid.UUID, payload: AudioRenamePayload) -> AudioFileListItem:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio name is required")
    try:
        with database_session() as session:
            item = audio_crud.get_audio_file(session, audio_file_id)
            updated = audio_crud.update_audio_file(
                session,
                audio_file_id,
                AudioUpdate(
                    name=name,
                    wav_bytes=None,
                    duration=item.duration,
                    segments=item.segments,
                    annotations=audio_annotations(item),
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.patch("/{audio_file_id}/score", response_model=AudioFileListItem)
async def update_audio_score(audio_file_id: uuid.UUID, payload: AudioScorePayload) -> AudioFileListItem:
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
                    annotations=audio_annotations(item).model_copy(update={"score": payload.score}),
                    segments=item.segments,
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.patch("/{audio_file_id}/language", response_model=AudioFileListItem)
async def update_audio_language(audio_file_id: uuid.UUID, payload: AudioLanguagePayload) -> AudioFileListItem:
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
                    language=payload.language,
                    segments=item.segments,
                    annotations=audio_annotations(item),
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.patch("/{audio_file_id}/style-prompt", response_model=AudioFileListItem)
async def update_audio_style_prompt(audio_file_id: uuid.UUID, payload: AudioStylePromptPayload) -> AudioFileListItem:
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
                    style_prompt=payload.style_prompt,
                    segments=item.segments,
                    annotations=audio_annotations(item),
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.patch("/{audio_file_id}/voice-prompt", response_model=AudioFileListItem)
async def update_audio_voice_prompt(audio_file_id: uuid.UUID, payload: AudioVoicePromptPayload) -> AudioFileListItem:
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
                    voice_prompt=payload.voice_prompt,
                    segments=item.segments,
                    annotations=audio_annotations(item),
                    virtual=item.virtual,
                ),
            )
            return audio_response(updated, None)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


router.include_router(content_router)
