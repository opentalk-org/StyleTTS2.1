import uuid
from typing import Any

from fastapi import APIRouter, Query

from backend.audio.schemas import AudioFileListItem, AudioFilePage, AudioSegmentRead, AudioSort
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile


router = APIRouter(prefix="/audio-files", tags=["audio-files"])


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
        return AudioFilePage(rows=[audio_response(item) for item in rows], total=total)


def audio_response(item: AudioFile) -> AudioFileListItem:
    metadata = dict(item.metadata_)
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        speaker=_speaker(metadata),
        duration=item.duration,
        sample_rate=_sample_rate(metadata),
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=len(item.segments),
        segment_preview=[segment_response(segment) for segment in item.segments[:8]],
        dataset_ids=[dataset.id for dataset in item.datasets],
        virtual=item.virtual,
        metadata=metadata,
        updated_at=item.updated_at,
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


def _segment_speaker(segment: dict[str, Any]) -> str:
    if "speaker" in segment:
        return str(segment["speaker"])
    if "voice" in segment:
        return str(segment["voice"])
    if "voice_id" in segment and segment["voice_id"] is not None:
        return str(uuid.UUID(str(segment["voice_id"])))
    return "-"
