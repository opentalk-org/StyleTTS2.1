from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.audio.pack_store import ObjectStore
from shared.db.audio.catalog import get_audio_files_bulk
from shared.db.audio.ranges.cache import AudioFileCache, StoredWavLocation
from shared.db.audio.ranges.types import SegmentReadRequest
from shared.db.audio.ranges.wav import WavClip, WavTimeRange, slice_wav_ranges


class BulkWavReader:
    def __init__(
        self,
        store: ObjectStore,
        cache: AudioFileCache | None,
        worker_count: int,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("bulk WAV worker count must be positive")
        self.store = store
        self.cache = cache
        self.executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="wav-fetch",
        )

    def read(
        self,
        session: Session,
        requests: tuple[SegmentReadRequest, ...],
    ) -> list[WavClip]:
        if not requests:
            return []
        audio_ids = tuple(
            dict.fromkeys(request.audio_file_id for request in requests)
        )
        items = get_audio_files_bulk(session, audio_ids)
        locations = {
            audio_id: self._location(items[audio_id]) for audio_id in audio_ids
        }
        futures: dict[UUID, Future[bytes]] = {
            audio_id: self.executor.submit(self._load, location)
            for audio_id, location in locations.items()
        }
        wavs = {
            audio_id: future.result() for audio_id, future in futures.items()
        }
        if self.cache is not None:
            self.cache.enforce_budget()
        grouped: dict[UUID, list[tuple[int, SegmentReadRequest]]] = defaultdict(list)
        for index, request in enumerate(requests):
            grouped[request.audio_file_id].append((index, request))
        output: list[WavClip | None] = [None] * len(requests)
        for audio_id, indexed in grouped.items():
            ranges = [
                WavTimeRange(request.start, request.end)
                for _, request in indexed
            ]
            clips = slice_wav_ranges(wavs[audio_id], ranges)
            for (index, _), clip in zip(indexed, clips, strict=True):
                output[index] = clip
        if any(clip is None for clip in output):
            raise RuntimeError("bulk WAV reader did not produce every clip")
        return [clip for clip in output if clip is not None]

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _load(self, location: StoredWavLocation) -> bytes:
        if self.cache is None:
            return self.store.read_range(
                location.object_path,
                location.byte_offset,
                location.byte_length,
            )
        return self.cache.load(
            location,
            lambda: self.store.read_range(
                location.object_path,
                location.byte_offset,
                location.byte_length,
            ),
        )

    @staticmethod
    def _location(item) -> StoredWavLocation:
        if item.storage_kind != "packed":
            raise ValueError(
                f"Audio {item.id} contains metadata only; "
                "no stored WAV bytes are available"
            )
        if item.bucket_file is None:
            raise ValueError(f"packed audio has no bucket: {item.id}")
        return StoredWavLocation(
            item.id,
            item.bucket_file.path,
            item.byte_offset,
            item.byte_length,
        )
