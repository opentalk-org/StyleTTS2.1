import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.audio.content_api import router as content_router
from backend.audio.operations import (
    delete_audio_records,
    full_audio_response,
    persist_uploaded_audio,
    required_audio,
    sample_rate,
    segment_record,
    update_audio,
)
from backend.audio.responses import (
    audio_list_response,
    audio_payload,
    segment_response,
)
from backend.audio.schemas import (
    AddToDatasetRequest,
    AudioFileListItem,
    AudioFilePage,
    AudioLanguagePayload,
    AudioRenamePayload,
    AudioScorePayload,
    AudioSegmentRead,
    AudioSegmentWrite,
    AudioSort,
    AudioStylePromptPayload,
    AudioVoicePromptPayload,
)
from shared.db.audio import clickhouse as audio
from shared.db.audio.catalog_pagination import AudioCursor, cursor_for_row
from shared.db.datasets import clickhouse as datasets

router = APIRouter(prefix="/audio-files", tags=["audio-files"])


@router.get("", response_model=AudioFilePage)
def list_audio_files(
    query: str = "",
    language: str = "",
    dataset: str = "all",
    sort: AudioSort = "updated",
    limit: int = Query(100, ge=1, le=200),
    cursor: str | None = None,
) -> AudioFilePage:
    try:
        page_cursor = AudioCursor.decode(cursor, sort) if cursor else None
        rows = audio.list_audio_files(
            limit=limit + 1,
            order=sort,
            after_value=page_cursor.value if page_cursor else None,
            after_id=page_cursor.audio_file_id if page_cursor else None,
            dataset=dataset,
            query=query,
            language=language,
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    has_more = len(rows) > limit
    rows = rows[:limit]
    ids = [item.id for item in rows]
    counts = audio.count_audio_segments(ids)
    previews = audio.list_audio_segment_previews(ids, 8)
    memberships = datasets.dataset_ids_by_audio_file(ids)
    response_rows = [
        audio_list_response(
            item,
            counts[item.id],
            sample_rate(item),
            previews[item.id],
            memberships[item.id],
        )
        for item in rows
    ]
    next_cursor = cursor_for_row(sort, rows[-1]).encode() if has_more else None
    return AudioFilePage(rows=response_rows, next_cursor=next_cursor, has_more=has_more)


@router.post(
    "/upload", response_model=AudioFileListItem, status_code=status.HTTP_201_CREATED
)
async def upload_audio_file(
    file: UploadFile = File(),
    duration: float = Form(),
    sample_rate: int = Form(),
    dataset_id: str = Form(""),
) -> AudioFileListItem:
    data = await file.read()
    if not data or duration <= 0 or sample_rate <= 0 or file.filename is None:
        raise HTTPException(status_code=400, detail="Valid non-empty audio is required")
    payload = audio_payload(file, data, duration, sample_rate)
    try:
        return await run_in_threadpool(persist_uploaded_audio, payload, dataset_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/dataset-membership", status_code=204)
async def add_audio_files_to_dataset(payload: AddToDatasetRequest) -> None:
    ids = (
        audio.search_audio_file_ids(payload.query, payload.dataset, payload.language)
        if payload.mode == "filter"
        else payload.audio_file_ids
    )
    now = datetime.now(UTC)
    try:
        datasets.add_audio_files(payload.dataset_id, ids, now, now)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("", status_code=204)
async def delete_matching_audio_files(
    query: str = "", language: str = "", dataset: str = Query(min_length=1)
) -> None:
    try:
        delete_audio_records(audio.search_audio_file_ids(query, dataset, language))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/{audio_file_id}", status_code=204)
async def delete_audio_file(audio_file_id: uuid.UUID) -> None:
    required_audio(audio_file_id)
    delete_audio_records([audio_file_id])


@router.get("/by-run/{run_id}", response_model=list[AudioFileListItem])
async def list_audio_files_by_run(run_id: str) -> list[AudioFileListItem]:
    return [full_audio_response(item) for item in audio.list_audio_files_by_run(run_id)]


@router.get("/{audio_file_id}/segment-preview", response_model=list[AudioSegmentRead])
def get_audio_segment_preview(
    audio_file_id: uuid.UUID, limit: int = Query(8, ge=1, le=32)
) -> list[AudioSegmentRead]:
    return [
        segment_response(item)
        for item in audio.list_audio_segment_previews([audio_file_id], limit)[
            audio_file_id
        ]
    ]


@router.get("/{audio_file_id}", response_model=AudioFileListItem)
async def get_audio_file(audio_file_id: uuid.UUID) -> AudioFileListItem:
    return full_audio_response(required_audio(audio_file_id))


@router.put("/{audio_file_id}/segments", response_model=AudioFileListItem)
async def replace_audio_segments(
    audio_file_id: uuid.UUID, payload: list[AudioSegmentWrite]
) -> AudioFileListItem:
    item = required_audio(audio_file_id)
    now = datetime.now(UTC)
    audio.replace_audio_segments(
        audio_file_id,
        [
            segment_record(audio_file_id, index, segment, now)
            for index, segment in enumerate(payload)
        ],
    )
    return full_audio_response(update_audio(item, updated_at=now))


@router.patch("/{audio_file_id}/name", response_model=AudioFileListItem)
async def rename_audio_file(
    audio_file_id: uuid.UUID, payload: AudioRenamePayload
) -> AudioFileListItem:
    return full_audio_response(
        update_audio(
            required_audio(audio_file_id),
            name=payload.name.strip(),
            updated_at=datetime.now(UTC),
        )
    )


@router.patch("/{audio_file_id}/score", response_model=AudioFileListItem)
async def update_audio_score(
    audio_file_id: uuid.UUID, payload: AudioScorePayload
) -> AudioFileListItem:
    return full_audio_response(
        update_audio(
            required_audio(audio_file_id),
            score=payload.score,
            updated_at=datetime.now(UTC),
        )
    )


@router.patch("/{audio_file_id}/language", response_model=AudioFileListItem)
async def update_audio_language(
    audio_file_id: uuid.UUID, payload: AudioLanguagePayload
) -> AudioFileListItem:
    return full_audio_response(
        update_audio(
            required_audio(audio_file_id),
            language=payload.language,
            updated_at=datetime.now(UTC),
        )
    )


@router.patch("/{audio_file_id}/style-prompt", response_model=AudioFileListItem)
async def update_audio_style_prompt(
    audio_file_id: uuid.UUID, payload: AudioStylePromptPayload
) -> AudioFileListItem:
    return full_audio_response(
        update_audio(
            required_audio(audio_file_id),
            style_prompt=payload.style_prompt,
            updated_at=datetime.now(UTC),
        )
    )


@router.patch("/{audio_file_id}/voice-prompt", response_model=AudioFileListItem)
async def update_audio_voice_prompt(
    audio_file_id: uuid.UUID, payload: AudioVoicePromptPayload
) -> AudioFileListItem:
    return full_audio_response(
        update_audio(
            required_audio(audio_file_id),
            voice_prompt=payload.voice_prompt,
            updated_at=datetime.now(UTC),
        )
    )


router.include_router(content_router)
