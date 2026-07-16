from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.audio.pack_crud import bulk_read_packed_audio_files
from shared.db.audio.pack_store import ObjectStore
from shared.db.audio.ranges.wav import (
    WavClip,
    WavTimeRange,
    read_wav_ranges,
    slice_wav_ranges,
)
from shared.db.audio.rows_crud import get_audio_files_bulk
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore


@dataclass(frozen=True)
class SegmentReadRequest:
    audio_file_id: UUID
    start: float
    end: float


def bulk_read_wav_segments(
    session: Session,
    requests: list[SegmentReadRequest],
    maximum_full_read_bytes: int,
    store: ObjectStore | None = None,
) -> list[WavClip]:
    if not requests:
        return []
    resolved_store = store or S3ObjectStore(settings_crud.object_store_config(session))
    items = get_audio_files_bulk(
        session,
        [request.audio_file_id for request in requests],
    )
    request_groups: dict[UUID, list[tuple[int, SegmentReadRequest]]] = defaultdict(list)
    for index, request in enumerate(requests):
        request_groups[request.audio_file_id].append((index, request))

    results: list[WavClip | None] = [None] * len(requests)
    full_pack_groups: dict[UUID, list[UUID]] = defaultdict(list)
    ranged_ids = []
    for audio_file_id, item in items.items():
        if item.storage_kind != "packed":
            raise ValueError(
                f"Audio {audio_file_id} contains metadata only; no stored WAV bytes are available"
            )
        assert item.bucket_file is not None, f"packed audio has no bucket: {audio_file_id}"
        if item.bucket_file.size <= maximum_full_read_bytes:
            full_pack_groups[item.bucket_file.id].append(audio_file_id)
        else:
            ranged_ids.append(audio_file_id)

    for audio_ids in full_pack_groups.values():
        stored = bulk_read_packed_audio_files(session, resolved_store, audio_ids)
        for audio_file_id in audio_ids:
            _store_clips(
                results,
                request_groups[audio_file_id],
                slice_wav_ranges,
                stored[audio_file_id],
            )
    for audio_file_id in ranged_ids:
        item = items[audio_file_id]
        assert item.bucket_file is not None, f"packed audio has no bucket: {audio_file_id}"
        indexed = request_groups[audio_file_id]
        ranges = [WavTimeRange(request.start, request.end) for _, request in indexed]
        clips = read_wav_ranges(
            resolved_store,
            item.bucket_file.path,
            item.byte_offset,
            item.byte_length,
            ranges,
        )
        for (index, _), clip in zip(indexed, clips, strict=True):
            results[index] = clip

    assert all(result is not None for result in results), "missing stored WAV segment result"
    return [result for result in results if result is not None]


def _store_clips(
    results: list[WavClip | None],
    indexed: list[tuple[int, SegmentReadRequest]],
    slicer,
    data: bytes,
) -> None:
    ranges = [WavTimeRange(request.start, request.end) for _, request in indexed]
    clips = slicer(data, ranges)
    for (index, _), clip in zip(indexed, clips, strict=True):
        results[index] = clip
