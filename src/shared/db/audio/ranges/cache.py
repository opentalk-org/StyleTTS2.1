import fcntl
import hashlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class StoredWavLocation:
    audio_file_id: UUID
    object_path: str
    byte_offset: int
    byte_length: int

    def __post_init__(self) -> None:
        if not self.object_path:
            raise ValueError("stored WAV object path must not be empty")
        if self.byte_offset < 0 or self.byte_length <= 0:
            raise ValueError("stored WAV byte range must be positive")

    @property
    def cache_key(self) -> str:
        payload = (
            f"{self.object_path}\0{self.byte_offset}\0{self.byte_length}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class AudioFileCache:
    def __init__(self, root: Path, budget_bytes: int) -> None:
        if budget_bytes <= 0:
            raise ValueError("audio cache budget must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = budget_bytes
        self._thread_lock = threading.Lock()

    def load(
        self,
        location: StoredWavLocation,
        fetch: Callable[[], bytes],
    ) -> bytes:
        path = self.root / f"{location.cache_key}.wav"
        lock_path = self.root / f"{location.cache_key}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self._read_valid(path, location.byte_length)
            if data is None:
                data = fetch()
                if len(data) != location.byte_length:
                    raise EOFError(
                        f"audio {location.audio_file_id} returned {len(data)} bytes; "
                        f"expected {location.byte_length}"
                    )
                self._write(path, data)
            else:
                os.utime(path, None)
        self._evict(path)
        return data

    def invalidate(self, location: StoredWavLocation) -> None:
        path = self.root / f"{location.cache_key}.wav"
        lock_path = self.root / f"{location.cache_key}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path.unlink(missing_ok=True)

    @staticmethod
    def _read_valid(path: Path, expected_bytes: int) -> bytes | None:
        if not path.is_file():
            return None
        data = path.read_bytes()
        if len(data) == expected_bytes:
            return data
        path.unlink()
        return None

    def _write(self, path: Path, data: bytes) -> None:
        temporary = self.root / (
            f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    def _evict(self, protected: Path) -> None:
        global_path = self.root / ".eviction.lock"
        with self._thread_lock, global_path.open("a+b") as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX)
            entries = sorted(
                self.root.glob("*.wav"),
                key=lambda item: item.stat().st_mtime_ns,
            )
            total = sum(item.stat().st_size for item in entries)
            for victim in entries:
                if total <= self.budget_bytes:
                    break
                if victim == protected:
                    continue
                lock_path = victim.with_suffix(".lock")
                with lock_path.open("a+b") as victim_lock:
                    try:
                        fcntl.flock(
                            victim_lock,
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        continue
                    size = victim.stat().st_size
                    victim.unlink()
                    total -= size
