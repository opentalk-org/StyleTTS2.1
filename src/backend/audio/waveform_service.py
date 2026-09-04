import asyncio
import multiprocessing
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

from shared.db import database_session
from shared.db.audio.clickhouse import get_audio_file as get_audio_record
from shared.db.audio.storage_locations import audio_storage_locations
from shared.db.settings import crud as settings_crud
from shared.db.waveforms import crud as waveform_crud
from shared.db.waveforms.codec import waveform_from_audio_bytes
from shared.storage import ObjectRange
from shared.logging_setup import get_logger

logger = get_logger("backend.audio.waveform")


class WaveformService:
    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self._pool: ProcessPoolExecutor | None = None
        self._pending: set[uuid.UUID] = set()
        self._failed: set[uuid.UUID] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    async def ensure(
        self, audio_file_id: uuid.UUID
    ) -> Literal["ready", "pending", "error"]:
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(None, self._exists, audio_file_id):
            return "ready"
        async with self._lock:
            if audio_file_id in self._pending:
                return "pending"
            if audio_file_id in self._failed:
                return "error"
            self._pending.add(audio_file_id)
        task = loop.create_task(self._generate(audio_file_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return "pending"

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    async def _generate(self, audio_file_id: uuid.UUID) -> None:
        loop = asyncio.get_running_loop()
        try:
            audio_bytes, duration = await loop.run_in_executor(
                None, self._read_audio, audio_file_id
            )
            waveform = await loop.run_in_executor(
                self._get_pool(), waveform_from_audio_bytes, audio_bytes
            )
            await loop.run_in_executor(
                None, self._write_waveform, audio_file_id, duration, waveform
            )
        except Exception:
            logger.exception(
                "waveform generation failed audio_file_id=%s", audio_file_id
            )
            async with self._lock:
                self._failed.add(audio_file_id)
        finally:
            async with self._lock:
                self._pending.discard(audio_file_id)

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def _exists(self, audio_file_id: uuid.UUID) -> bool:
        return waveform_crud.waveform_exists(audio_file_id)

    def _read_audio(self, audio_file_id: uuid.UUID) -> tuple[bytes, float]:
        with database_session() as session:
            item = get_audio_record(audio_file_id)
            location = audio_storage_locations(session, [audio_file_id])[audio_file_id]
            data = settings_crud.object_store(session).read_range(
                ObjectRange(
                    location.object_path,
                    location.byte_offset,
                    location.byte_length,
                )
            )
            return data, item.duration

    def _write_waveform(
        self, audio_file_id: uuid.UUID, duration: float, waveform
    ) -> None:
        with database_session() as session:
            waveform_crud.replace_waveform(session, audio_file_id, duration, waveform)
