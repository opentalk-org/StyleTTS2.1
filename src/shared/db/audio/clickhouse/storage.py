import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from shared.db.assets.clickhouse import (
    BucketFileRecord,
    BucketKind,
    create_bucket_files,
)
from shared.db.assets.crud import delete_unreferenced_bucket_files
from shared.db.audio.clickhouse.conversion import segment_records
from shared.db.audio.clickhouse.files import (
    create_audio_files,
    get_audio_files,
)
from shared.db.audio.clickhouse.models import (
    AudioFileRecord,
    AudioFileUpdate,
    StorageKind,
)
from shared.db.audio.clickhouse.segments import replace_audio_segments_bulk
from shared.db.audio.schemas import AudioCreate, AudioPartRead, AudioUpdate
from shared.db.audio.storage_locations import audio_storage_locations
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.storage import ObjectRange, S3RequestMetrics


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    target_pack_bytes: int = 128 * 1024 * 1024,
    path_prefix: str = "audio-packs",
) -> list[AudioFileRecord]:
    if not payloads:
        return []
    now = datetime.now(UTC)
    store = settings_crud.object_store(session)
    packs: list[BucketFileRecord] = []
    records: list[AudioFileRecord] = []
    for pack_payloads in _payload_packs(payloads, target_pack_bytes):
        bucket_id = uuid4()
        path = f"{path_prefix}/{bucket_id}.bin"
        data = b"".join(payload.wav_bytes for payload in pack_payloads)
        store.upload(path, data)
        packs.append(
            BucketFileRecord(
                id=bucket_id,
                kind=BucketKind.AUDIO,
                path=path,
                size=len(data),
            )
        )
        byte_offset = 0
        for payload in pack_payloads:
            records.append(_record(uuid4(), bucket_id, byte_offset, payload, now))
            byte_offset += len(payload.wav_bytes)
    create_bucket_files(packs)
    create_audio_files(records)
    replace_audio_segments_bulk(
        {
            record.id: segment_records(record.id, payload.segments, now)
            for record, payload in zip(records, payloads, strict=True)
        }
    )
    for record, payload in zip(records, payloads, strict=True):
        if payload.waveform is not None:
            waveform_crud.replace_waveform(
                session, record.id, record.duration, payload.waveform
            )
    return records


def bulk_update_audio_files(
    session: Session,
    payloads: dict[UUID, AudioUpdate],
    target_pack_bytes: int = 128 * 1024 * 1024,
    path_prefix: str = "audio-packs",
) -> dict[UUID, AudioFileRecord]:
    current = {item.id: item for item in get_audio_files(list(payloads))}
    missing = set(payloads).difference(current)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(map(str, missing))}")
    now = datetime.now(UTC)
    latest = max(item.updated_at for item in current.values())
    if now <= latest:
        now = latest + timedelta(microseconds=1)
    updated: list[AudioFileRecord] = []
    store = settings_crud.object_store(session)
    replacement_locations: dict[UUID, tuple[UUID, int]] = {}
    replacement_packs = []
    for pack_payloads in _update_payload_packs(payloads, target_pack_bytes):
        replacement_bucket_id = uuid4()
        path = f"{path_prefix}/{replacement_bucket_id}.bin"
        data = b"".join(payload.wav_bytes or b"" for _, payload in pack_payloads)
        store.upload(path, data)
        replacement_packs.append(
            BucketFileRecord(
                id=replacement_bucket_id,
                kind=BucketKind.AUDIO,
                path=path,
                size=len(data),
            )
        )
        byte_offset = 0
        for audio_id, payload in pack_payloads:
            replacement_locations[audio_id] = (
                replacement_bucket_id,
                byte_offset,
            )
            byte_offset += len(payload.wav_bytes or b"")
    create_bucket_files(replacement_packs)
    for audio_id, payload in payloads.items():
        item = current[audio_id]
        bucket_id = item.bucket_file_id
        byte_length = item.byte_length
        byte_offset = item.byte_offset
        if payload.wav_bytes is not None:
            bucket_id, byte_offset = replacement_locations[audio_id]
            byte_length = len(payload.wav_bytes)
        updated.append(
            _updated_record(item, payload, bucket_id, byte_offset, byte_length, now)
        )
    create_audio_files(updated)
    replace_audio_segments_bulk(
        {
            record.id: segment_records(record.id, payload.segments, now)
            for record, payload in zip(updated, payloads.values(), strict=True)
        }
    )
    for record, payload in zip(updated, payloads.values(), strict=True):
        if payload.wav_bytes is not None:
            waveform_crud.bulk_delete_waveforms(session, [record.id])
        if payload.waveform is not None:
            waveform_crud.replace_waveform(
                session, record.id, record.duration, payload.waveform
            )
    replaced_bucket_ids = []
    for audio_id, payload in payloads.items():
        bucket_file_id = current[audio_id].bucket_file_id
        if payload.wav_bytes is not None and bucket_file_id is not None:
            replaced_bucket_ids.append(bucket_file_id)
    delete_unreferenced_bucket_files(session, replaced_bucket_ids)
    return {item.id: item for item in updated}


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[UUID],
    request_metrics: S3RequestMetrics | None = None,
) -> dict[UUID, bytes]:
    ids = list(dict.fromkeys(audio_file_ids))
    locations = audio_storage_locations(session, ids)
    ranges = [
        ObjectRange(
            locations[item].object_path,
            locations[item].byte_offset,
            locations[item].byte_length,
        )
        for item in ids
    ]
    started = time.monotonic()
    payloads = settings_crud.object_store(session, request_metrics).read_ranges(ranges)
    if request_metrics is not None:
        request_metrics.fetch_seconds = time.monotonic() - started
        request_metrics.fetch_bytes = sum(map(len, payloads))
    return dict(zip(ids, payloads, strict=True))


def read_audio_part(
    session: Session, audio_file_id: UUID, payload: AudioPartRead
) -> bytes:
    location = audio_storage_locations(session, [audio_file_id])[audio_file_id]
    if (
        payload.start < 0
        or payload.length <= 0
        or payload.start + payload.length > location.byte_length
    ):
        raise ValueError(f"invalid audio byte range: {audio_file_id}")
    return settings_crud.object_store(session).read_range(
        ObjectRange(
            location.object_path, location.byte_offset + payload.start, payload.length
        )
    )


def _payload_packs(
    payloads: Sequence[AudioCreate], target_pack_bytes: int
) -> list[list[AudioCreate]]:
    assert target_pack_bytes > 0, "target pack size must be positive"
    packs: list[list[AudioCreate]] = []
    current: list[AudioCreate] = []
    current_bytes = 0
    for payload in payloads:
        if current and current_bytes + len(payload.wav_bytes) > target_pack_bytes:
            packs.append(current)
            current = []
            current_bytes = 0
        current.append(payload)
        current_bytes += len(payload.wav_bytes)
    if current:
        packs.append(current)
    return packs


def _update_payload_packs(
    payloads: dict[UUID, AudioUpdate], target_pack_bytes: int
) -> list[list[tuple[UUID, AudioUpdate]]]:
    assert target_pack_bytes > 0, "target pack size must be positive"
    packs: list[list[tuple[UUID, AudioUpdate]]] = []
    current: list[tuple[UUID, AudioUpdate]] = []
    current_bytes = 0
    for audio_id, payload in payloads.items():
        if payload.wav_bytes is None:
            continue
        if current and current_bytes + len(payload.wav_bytes) > target_pack_bytes:
            packs.append(current)
            current = []
            current_bytes = 0
        current.append((audio_id, payload))
        current_bytes += len(payload.wav_bytes)
    if current:
        packs.append(current)
    return packs


def _record(
    audio_id: UUID,
    bucket_id: UUID,
    byte_offset: int,
    payload: AudioCreate,
    now: datetime,
) -> AudioFileRecord:
    return AudioFileRecord(
        id=audio_id,
        updated_at=now,
        name=payload.name,
        bucket_file_id=bucket_id,
        byte_offset=byte_offset,
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


def _updated_record(
    item: AudioFileRecord,
    payload: AudioUpdate,
    bucket_id: UUID | None,
    byte_offset: int,
    byte_length: int,
    now: datetime,
) -> AudioFileRecord:
    update = AudioFileUpdate(
        name=payload.name,
        bucket_file_id=bucket_id,
        byte_offset=byte_offset,
        duration=payload.duration,
        byte_length=byte_length,
        score=payload.annotations.score,
        language=payload.language,
        style_prompt=payload.style_prompt,
        voice_prompt=payload.voice_prompt,
        virtual=payload.virtual,
        storage_kind=item.storage_kind,
        storage_ref=item.storage_ref,
        metadata=payload.annotations.metadata,
        updated_at=now,
    )
    return AudioFileRecord(id=item.id, **update.model_dump())
