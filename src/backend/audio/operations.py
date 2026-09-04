import uuid
from datetime import UTC, datetime

from fastapi import HTTPException, status

from backend.audio.responses import audio_response
from backend.audio.schemas import AudioFileListItem, AudioSegmentWrite
from shared.db import database_session
from shared.db.assets.clickhouse import (
    BucketFileRecord,
    BucketKind,
    create_bucket_files,
    delete_bucket_files,
)
from shared.db.assets.crud import delete_unreferenced_bucket_files
from shared.db.audio import clickhouse as audio
from shared.db.audio.clickhouse import (
    AudioFileRecord,
    AudioFileUpdate,
    AudioSegmentRecord,
    StorageKind,
)
from shared.db.audio.schemas import AudioCreate
from shared.db.datasets import crud as datasets
from shared.db.settings import crud as settings_crud
from shared.db.waveforms.clickhouse import get_waveforms


def persist_uploaded_audio(payload: AudioCreate, dataset_id: str) -> AudioFileListItem:
    audio_id = uuid.uuid4()
    pack_id = uuid.uuid4()
    path = f"audio-packs/{pack_id}.bin"
    parsed_dataset_id = uuid.UUID(dataset_id) if dataset_id else None
    if parsed_dataset_id is not None:
        datasets.get_dataset(parsed_dataset_id)
    with database_session() as session:
        settings_crud.object_store(session).upload(path, payload.wav_bytes)
    try:
        create_bucket_files(
            [
                BucketFileRecord(
                    id=pack_id,
                    kind=BucketKind.AUDIO,
                    path=path,
                    size=len(payload.wav_bytes),
                )
            ]
        )
        now = datetime.now(UTC)
        item = AudioFileRecord(
            id=audio_id,
            updated_at=now,
            name=payload.name,
            bucket_file_id=pack_id,
            byte_offset=0,
            duration=payload.duration,
            byte_length=len(payload.wav_bytes),
            score=payload.annotations.score,
            language=payload.language,
            style_prompt=payload.style_prompt,
            voice_prompt=payload.voice_prompt,
            virtual=payload.virtual,
            storage_kind=StorageKind.PACKED,
            storage_ref=None,
            metadata=payload.annotations.metadata,
        )
        audio.create_audio_files([item])
        memberships = []
        if parsed_dataset_id is not None:
            datasets.bulk_add_audio_files_to_dataset(parsed_dataset_id, [audio_id])
            memberships.append(parsed_dataset_id)
    except Exception:
        audio.delete_audio_files([audio_id])
        delete_bucket_files([pack_id])
        with database_session() as session:
            settings_crud.object_store(session).delete(path)
        raise
    return audio_response(item, [], memberships, None)


def update_audio(item: AudioFileRecord, **changes: object) -> AudioFileRecord:
    updated = item.model_copy(update=changes)
    payload = AudioFileUpdate.model_validate(updated.model_dump())
    return audio.update_audio_file(item.id, payload)


def delete_audio_records(audio_file_ids: list[uuid.UUID]) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    files = audio.get_audio_files(ids)
    waveforms = get_waveforms(ids)
    bucket_file_ids = [
        item.bucket_file_id for item in files if item.bucket_file_id is not None
    ]
    bucket_file_ids.extend(item.pack_id for item in waveforms)
    audio.delete_audio_files(ids)
    with database_session() as session:
        delete_unreferenced_bucket_files(session, bucket_file_ids)


def full_audio_response(item: AudioFileRecord) -> AudioFileListItem:
    segments = audio.list_audio_segments(item.id)
    memberships = datasets.dataset_ids_by_audio_file([item.id])
    return audio_response(item, segments, memberships[item.id], None)


def segment_record(
    audio_file_id: uuid.UUID,
    position: int,
    item: AudioSegmentWrite,
    updated_at: datetime,
) -> AudioSegmentRecord:
    return AudioSegmentRecord(
        id=item.id,
        audio_file_id=audio_file_id,
        updated_at=updated_at,
        position=position,
        start_seconds=item.start,
        end_seconds=item.end,
        text=item.text,
        phon=item.phon,
        kind=item.type_,
        accuracy=item.annotations.accuracy,
        speaker_id=item.annotations.speaker_id,
        metadata=item.annotations.metadata,
        alignment=[value.model_dump() for value in item.alignment]
        if item.alignment
        else None,
    )


def required_audio(audio_file_id: uuid.UUID) -> AudioFileRecord:
    try:
        return audio.get_audio_file(audio_file_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


def sample_rate(item: AudioFileRecord) -> int | None:
    value = item.metadata.get("sample_rate")
    return int(value) if value is not None else None
