from typing import Any
from uuid import UUID

from fastapi import HTTPException, UploadFile, status

from backend.audio.schemas import (
    AudioFileListItem,
    AudioSegmentRead,
    WordAlignment,
)
from shared.audio_annotations import AudioAnnotations
from shared.db.audio.clickhouse import AudioFileRecord, AudioSegmentRecord, StorageKind
from shared.db.audio.schemas import AudioCreate

DEFAULT_STREAM_CHUNK = 1024 * 1024


def audio_response(
    item: AudioFileRecord,
    segments: list[AudioSegmentRecord],
    dataset_ids: list[UUID],
    segment_limit: int | None,
) -> AudioFileListItem:
    preview = segments if segment_limit is None else segments[:segment_limit]
    return _audio_item(item, len(segments), preview, dataset_ids)


def audio_list_response(
    item: AudioFileRecord,
    segment_count: int,
    preview_sample_rate: int | None,
    segment_preview: list[AudioSegmentRecord],
    dataset_ids: list[UUID],
) -> AudioFileListItem:
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        annotations=AudioAnnotations(
            score=item.score,
            accuracy=None,
            metadata=item.metadata,
        ),
        duration=item.duration,
        language=item.language,
        style_prompt=item.style_prompt,
        voice_prompt=item.voice_prompt,
        sample_rate=preview_sample_rate,
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=segment_count,
        segment_preview=[segment_response(segment) for segment in segment_preview],
        dataset_ids=dataset_ids,
        virtual=item.virtual,
        storage_kind=item.storage_kind.value,
        updated_at=item.updated_at,
    )


def audio_payload(
    file: UploadFile,
    data: bytes,
    duration: float,
    sample_rate: int,
) -> AudioCreate:
    assert file.filename is not None, "audio filename is required"
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


def segment_response(segment: AudioSegmentRecord) -> AudioSegmentRead:
    return AudioSegmentRead(
        id=segment.id,
        start=segment.start_seconds,
        end=segment.end_seconds,
        text=segment.text,
        phon=segment.phon,
        annotations=AudioAnnotations(
            speaker_id=segment.speaker_id,
            accuracy=segment.accuracy,
            metadata=segment.metadata,
        ),
        type_=segment.kind,
        alignment=_segment_alignment(segment.alignment),
    )


def audio_annotations(
    item: AudioFileRecord,
    segments: list[AudioSegmentRecord],
) -> AudioAnnotations:
    speakers = {segment.speaker_id for segment in segments if segment.speaker_id}
    return AudioAnnotations(
        speaker_id=speakers.pop() if len(speakers) == 1 else None,
        score=item.score,
        accuracy=None,
        metadata=item.metadata,
    )


def sample_rate(metadata: dict[str, Any]) -> int | None:
    if "sample_rate" not in metadata:
        return None
    return int(metadata["sample_rate"])


def content_type(metadata: dict[str, Any]) -> str:
    if "content_type" in metadata:
        return str(metadata["content_type"])
    return "application/octet-stream"


def require_packed_audio(item: AudioFileRecord) -> None:
    if item.storage_kind != StorageKind.PACKED:
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
    item: AudioFileRecord,
    segment_count: int,
    segment_preview: list[AudioSegmentRecord],
    dataset_ids: list[UUID],
) -> AudioFileListItem:
    metadata = item.metadata
    return AudioFileListItem(
        id=item.id,
        name=item.name,
        annotations=audio_annotations(item, segment_preview),
        duration=item.duration,
        language=item.language,
        style_prompt=item.style_prompt,
        voice_prompt=item.voice_prompt,
        sample_rate=sample_rate(metadata),
        byte_length=item.byte_length,
        size_mb=f"{item.byte_length / 1024 / 1024:.1f}",
        segments=segment_count,
        segment_preview=[segment_response(segment) for segment in segment_preview],
        dataset_ids=dataset_ids,
        virtual=item.virtual,
        storage_kind=item.storage_kind.value,
        updated_at=item.updated_at,
    )


def _segment_alignment(raw: list[dict[str, Any]] | None) -> list[WordAlignment] | None:
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
