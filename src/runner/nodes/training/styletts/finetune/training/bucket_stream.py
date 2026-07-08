from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from runner.nodes.training.common.manifest.stream_plan import StreamPlan
from shared.db import database_session
from shared.db.audio import crud as audio_crud

logger = logging.getLogger(__name__)


class BucketStreamCache:
    """On-demand, budget-bounded local cache of training audio.

    Training visits samples in bucket order. This cache fetches whole bucket
    files (each downloaded once, only the needed slices unpacked to disk),
    prefetches ahead of the cursor as far as a storage budget allows, and keeps
    every already-consumed bucket resident until the budget forces LRU eviction
    of the oldest consumed one. Buckets at or ahead of the cursor are never
    evicted so in-flight samples always resolve."""

    def __init__(
        self,
        plan: StreamPlan,
        cache_dir: Path,
        budget_bytes: int,
        check_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._buckets = plan.buckets
        self._cache_dir = cache_dir.resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget = max(budget_bytes, 1)
        self._index_of = {audio_id: index for index, bucket in enumerate(self._buckets) for audio_id in bucket.audio_ids}
        self._resident: dict[int, int] = {}
        self._inflight: set[int] = set()
        self._cursor = 0
        self._stopped = False
        self._error: BaseException | None = None
        self._check_cancel = check_cancel
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._prefetch_loop, name="bucket-prefetch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def reset(self) -> None:
        """Rewind to the first bucket for a new epoch and drop the resident set.

        Each epoch replays the same bucket order, so the tail buckets left over
        from the previous epoch are the last thing needed again; clearing them
        lets prefetch refill from the front within budget."""
        with self._cond:
            for index in list(self._resident):
                self._evict(index)
            self._cursor = 0
            self._cond.notify_all()

    def bucket_index_of(self, audio_id: UUID) -> int:
        return self._index_of[audio_id]

    def advance(self, bucket_index: int) -> None:
        with self._cond:
            if bucket_index > self._cursor:
                self._cursor = bucket_index
                self._cond.notify_all()

    def resident_audio_ids(self) -> list[UUID]:
        with self._cond:
            indices = list(self._resident)
        return [audio_id for index in indices for audio_id in self._buckets[index].audio_ids]

    def ensure(self, audio_id: UUID) -> Path:
        index = self._index_of[audio_id]
        while True:
            with self._cond:
                self._raise_if_error()
                if index in self._resident:
                    return self._wav_path(audio_id)
                if index in self._inflight:
                    self._cond.wait()
                    continue
                self._inflight.add(index)
            self._fetch(index)
            return self._wav_path(audio_id)

    def _prefetch_loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and self._error is None and self._pick_prefetch() is None:
                    self._cond.wait()
                if self._stopped or self._error is not None:
                    return
                index = self._pick_prefetch()
                self._inflight.add(index)
            try:
                self._fetch(index)
            except BaseException:
                return

    def _pick_prefetch(self) -> int | None:
        resident_bytes = sum(self._resident.values())
        evictable_bytes = sum(size for index, size in self._resident.items() if index < self._cursor)
        for index in range(self._cursor, len(self._buckets)):
            if index in self._resident or index in self._inflight:
                continue
            size = self._buckets[index].byte_length
            if index == self._cursor or resident_bytes - evictable_bytes + size <= self._budget:
                return index
            return None
        return None

    def _fetch(self, index: int) -> None:
        bucket = self._buckets[index]
        try:
            self._run_cancel_check()
            with self._cond:
                self._make_room(bucket.byte_length, index)
            with database_session() as session:
                wavs = audio_crud.bulk_read_audio_files(session, bucket.audio_ids)
            written = 0
            for audio_id, wav_bytes in wavs.items():
                self._wav_path(audio_id).write_bytes(wav_bytes)
                written += len(wav_bytes)
            with self._cond:
                self._resident[index] = written
                self._inflight.discard(index)
                self._cond.notify_all()
        except BaseException as exc:
            with self._cond:
                self._error = exc
                self._inflight.discard(index)
                self._cond.notify_all()
            raise

    def _make_room(self, needed: int, protect_index: int) -> None:
        while sum(self._resident.values()) + needed > self._budget:
            consumed = [index for index in self._resident if index < self._cursor and index != protect_index]
            if not consumed:
                return
            victim = min(consumed)
            self._evict(victim)

    def _evict(self, index: int) -> None:
        for audio_id in self._buckets[index].audio_ids:
            self._wav_path(audio_id).unlink(missing_ok=True)
        del self._resident[index]
        logger.debug("evicted bucket %d from stream cache", index)

    def _run_cancel_check(self) -> None:
        if self._check_cancel is not None:
            self._check_cancel()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("bucket stream prefetch failed") from self._error

    def _wav_path(self, audio_id: UUID) -> Path:
        return self._cache_dir / f"{audio_id}.wav"
