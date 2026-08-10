from typing import Any

from fastapi import HTTPException, UploadFile, status

from backend.audio.schemas import (
    AudioFileListItem,
    AudioSegmentRead,
    WordAlignment,
)
from shared.audio_annotations import AudioAnnotations
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioCreate

DEFAULT_STREAM_CHUNK = 1024 * 1024


def audio_response(item: AudioFile, segment_limit: int | None) -> AudioFileListItem:
    segments = item.segments if segment_limit is None else item.segments[:segment_limit]
    return _audio_item(item, len(item.segments), segments)


def audio_list_response(
    item: AudioFile,
    segment_count: int,
    preview_sample_rate: int | None,
) -> AudioFileListItem:
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        annotations=AudioAnnotations(
            score=item.score,
            accuracy=None,
            metadata={},
        ),
        duration=item.duration,
        language=item.language,
        style_prompt=item.style_prompt,
        voice_prompt=item.voice_prompt,
        sample_rate=preview_sample_rate,
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=segment_count,
        segment_preview=[],
        dataset_ids=[dataset.id for dataset in item.datasets],
        virtual=item.virtual,
        storage_kind=item.storage_kind,
        updated_at=item.updated_at,
    )


def audio_payload(
    file: UploadFile,
    data: bytes,
    duration: float,
    sample_rate: int,
) -> AudioCreate:
    metadata: dict[str, Any] = {"sample_rate": sample_rate}
    if file.content_type is not None:
        metadata["content_type"] = file.content_type
    metadata["source_filename"] = file.filename
    return AudioCreate(
        name=file.filename,
        wav_bytes=data,
        duration=duration,
        annotations=AudioAnnotations(metadata=metadata),
        segments=[],
        virtual=False,
    )


def segment_response(segment: dict[str, Any]) -> AudioSegmentRead:
    return AudioSegmentRead(
        id=str(segment["id"]),
        start=float(segment["start"]),
        end=float(segment["end"]),
        text=str(segment["text"]),
        phon=str(segment["phon"]),
        annotations=AudioAnnotations.model_validate(segment["annotations"]),
        type_=str(segment["type_"]),
        alignment=_segment_alignment(segment),
    )


def audio_annotations(item: AudioFile) -> AudioAnnotations:
    return audio_crud.audio_file_annotations(item)


def sample_rate(metadata: dict[str, Any]) -> int | None:
    if "sample_rate" not in metadata:
        return None
    return int(metadata["sample_rate"])


def content_type(metadata: dict[str, Any]) -> str:
    if "content_type" in metadata:
        return str(metadata["content_type"])
    return "application/octet-stream"


def require_packed_audio(item: AudioFile) -> None:
    if item.storage_kind != "packed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Audio {item.id} contains metadata only; "
                "no stored audio bytes are available"
            ),
        )


def content_range(range_header: str | None, byte_length: int) -> tuple[int, int]:
    if range_header is None:
        return 0, min(byte_length, DEFAULT_STREAM_CHUNK) - 1
    unit, value = range_header.split("=", 1)
    if unit != "bytes":
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Only byte ranges are supported",
        )
    start_text, end_text = value.split("-", 1)
    start = int(start_text) if start_text else 0
    end = (
        int(end_text)
        if end_text
        else min(byte_length - 1, start + DEFAULT_STREAM_CHUNK - 1)
    )
    if start < 0 or end < start or start >= byte_length:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Requested range is invalid",
        )
    return start, min(end, byte_length - 1)


def _audio_item(
    item: AudioFile,
    segment_count: int,
    segment_preview: list[dict[str, Any]],
) -> AudioFileListItem:
    metadata = dict(item.metadata_)
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        annotations=audio_annotations(item),
        duration=item.duration,
        language=item.language,
        style_prompt=item.style_prompt,
        voice_prompt=item.voice_prompt,
        sample_rate=sample_rate(metadata),
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=segment_count,
        segment_preview=[
            segment_response(segment)
            for segment in segment_preview
            if all(field in segment for field in ("id", "type_", "alignment"))
        ],
        dataset_ids=[dataset.id for dataset in item.datasets],
        virtual=item.virtual,
        storage_kind=item.storage_kind,
        updated_at=item.updated_at,
    )


def _segment_alignment(segment: dict[str, Any]) -> list[WordAlignment] | None:
    raw = segment["alignment"]
    if raw is None:
        return None
    return [
        WordAlignment(
            word=str(item["word"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in raw
    ]
