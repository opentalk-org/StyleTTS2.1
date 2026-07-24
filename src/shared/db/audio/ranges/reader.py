from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.audio.ranges.wav import WavClip, WavTimeRange, slice_wav_ranges
from shared.db.audio.storage_locations import audio_storage_locations
from shared.db.settings import crud as settings_crud
from shared.storage import ObjectRange


@dataclass(frozen=True)
class SegmentReadRequest:
    audio_file_id: UUID
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("segment read range must be positive and ordered")


def bulk_read_wav_segments(
    session: Session,
    requests: list[SegmentReadRequest] | tuple[SegmentReadRequest, ...],
) -> list[WavClip]:
    if not requests:
        return []
    audio_ids = list(
        dict.fromkeys(request.audio_file_id for request in requests)
    )
    locations = audio_storage_locations(session, audio_ids)
    ranges = [
        ObjectRange(
            locations[audio_file_id].object_path,
            locations[audio_file_id].byte_offset,
            locations[audio_file_id].byte_length,
        )
        for audio_file_id in audio_ids
    ]
    payloads = settings_crud.object_store(session).read_ranges(ranges)
    wavs = dict(zip(audio_ids, payloads, strict=True))

    grouped: dict[UUID, list[tuple[int, SegmentReadRequest]]] = defaultdict(list)
    for index, request in enumerate(requests):
        grouped[request.audio_file_id].append((index, request))
    clips_by_index = {}
    for audio_file_id, indexed_requests in grouped.items():
        time_ranges = [
            WavTimeRange(request.start, request.end)
            for _, request in indexed_requests
        ]
        clips = slice_wav_ranges(wavs[audio_file_id], time_ranges)
        for (index, _), clip in zip(indexed_requests, clips, strict=True):
            clips_by_index[index] = clip
    return [clips_by_index[index] for index in range(len(requests))]
