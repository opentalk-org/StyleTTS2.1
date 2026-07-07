import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile


@dataclass(frozen=True)
class AudioPackConfig:
    target_pack_bytes: int = 128 * 1024 * 1024
    path_prefix: str = "audio-packs"
    prune_used_ratio: float = 0.5
    reuse_open_packs: bool = True
    seal_on_flush: bool = False


@dataclass(frozen=True)
class PackedWrite:
    bucket_file: BucketFile
    byte_offset: int
    byte_length: int


class ObjectStore(Protocol):
    def upload(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    def download(self, path: str) -> bytes:
        raise NotImplementedError

    def read_range(self, path: str, byte_offset: int, byte_length: int) -> bytes:
        raise NotImplementedError

    def delete(self, path: str) -> None:
        raise NotImplementedError


class AudioPackWriter:
    def __init__(self, session: Session, store: ObjectStore, config: AudioPackConfig) -> None:
        self._session = session
        self._store = store
        self._config = config
        self._active_pack: BucketFile | None = None
        self._pack_data: dict[uuid.UUID, bytearray] = {}
        self._packs: dict[uuid.UUID, BucketFile] = {}
        self._dirty_pack_ids: set[uuid.UUID] = set()

    def append(self, wav_bytes: bytes) -> PackedWrite:
        if len(wav_bytes) > self._config.target_pack_bytes:
            return self._write_oversized_pack(wav_bytes)
        pack = self._writable_pack(len(wav_bytes))
        data = self._active_pack_data(pack)
        byte_offset = pack.size
        data.extend(wav_bytes)
        pack.size += len(wav_bytes)
        pack.used_bytes += len(wav_bytes)
        pack.sealed = pack.size >= self._config.target_pack_bytes
        self._dirty_pack_ids.add(pack.id)
        return PackedWrite(bucket_file=pack, byte_offset=byte_offset, byte_length=len(wav_bytes))

    def flush(self) -> None:
        for pack_id in self._dirty_pack_ids:
            pack = self._packs[pack_id]
            self._store.upload(pack.path, bytes(self._pack_data[pack_id]))
            if self._config.seal_on_flush:
                pack.sealed = True

    def _write_oversized_pack(self, wav_bytes: bytes) -> PackedWrite:
        pack = self._create_pack(size=len(wav_bytes), used_bytes=len(wav_bytes), sealed=True)
        self._packs[pack.id] = pack
        self._pack_data[pack.id] = bytearray(wav_bytes)
        self._dirty_pack_ids.add(pack.id)
        return PackedWrite(bucket_file=pack, byte_offset=0, byte_length=len(wav_bytes))

    def _writable_pack(self, byte_length: int) -> BucketFile:
        if self._active_pack is not None and self._active_pack.size + byte_length <= self._config.target_pack_bytes:
            return self._active_pack
        pack = self._load_writable_pack(byte_length)
        self._active_pack = pack
        return pack

    def _load_writable_pack(self, byte_length: int) -> BucketFile:
        if not self._config.reuse_open_packs:
            return self._create_pack(size=0, used_bytes=0, sealed=False)
        statement = (
            select(BucketFile)
            .where(BucketFile.sealed.is_(False))
            .order_by(BucketFile.size.desc())
            .with_for_update()
        )
        for pack in self._session.execute(statement).scalars():
            if pack.size + byte_length <= self._config.target_pack_bytes:
                return pack
            pack.sealed = True
        return self._create_pack(size=0, used_bytes=0, sealed=False)

    def _create_pack(self, size: int, used_bytes: int, sealed: bool) -> BucketFile:
        pack = BucketFile(
            path=f"{self._config.path_prefix}/{uuid.uuid4()}.bin",
            size=size,
            used_bytes=used_bytes,
            sealed=sealed,
        )
        self._session.add(pack)
        self._session.flush()
        self._packs[pack.id] = pack
        return pack

    def _active_pack_data(self, pack: BucketFile) -> bytearray:
        self._packs[pack.id] = pack
        if pack.id not in self._pack_data:
            self._pack_data[pack.id] = bytearray()
            if pack.size > 0:
                self._pack_data[pack.id].extend(self._store.download(pack.path))
        return self._pack_data[pack.id]
