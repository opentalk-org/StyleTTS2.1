import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.waveforms.models import WaveformPack


@dataclass(frozen=True)
class WaveformPackConfig:
    target_pack_bytes: int = 64 * 1024 * 1024
    path_prefix: str = "waveform-packs"


@dataclass(frozen=True)
class WaveformWrite:
    pack: WaveformPack
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


class WaveformPackWriter:
    def __init__(self, session: Session, store: ObjectStore, config: WaveformPackConfig) -> None:
        self._session = session
        self._store = store
        self._config = config
        self._active_pack: WaveformPack | None = None
        self._pack_data: dict[uuid.UUID, bytearray] = {}
        self._packs: dict[uuid.UUID, WaveformPack] = {}
        self._dirty_pack_ids: set[uuid.UUID] = set()

    def append(self, data: bytes) -> WaveformWrite:
        if len(data) > self._config.target_pack_bytes:
            return self._write_oversized_pack(data)
        pack = self._writable_pack(len(data))
        pack_data = self._active_pack_data(pack)
        byte_offset = pack.size
        pack_data.extend(data)
        pack.size += len(data)
        pack.used_bytes += len(data)
        pack.sealed = pack.size >= self._config.target_pack_bytes
        self._dirty_pack_ids.add(pack.id)
        return WaveformWrite(pack=pack, byte_offset=byte_offset, byte_length=len(data))

    def flush(self) -> None:
        for pack_id in self._dirty_pack_ids:
            pack = self._packs[pack_id]
            self._store.upload(pack.path, bytes(self._pack_data[pack_id]))

    def _write_oversized_pack(self, data: bytes) -> WaveformWrite:
        pack = self._create_pack(size=len(data), used_bytes=len(data), sealed=True)
        self._pack_data[pack.id] = bytearray(data)
        self._dirty_pack_ids.add(pack.id)
        return WaveformWrite(pack=pack, byte_offset=0, byte_length=len(data))

    def _writable_pack(self, byte_length: int) -> WaveformPack:
        if self._active_pack is not None and self._active_pack.size + byte_length <= self._config.target_pack_bytes:
            return self._active_pack
        pack = self._load_writable_pack(byte_length)
        self._active_pack = pack
        return pack

    def _load_writable_pack(self, byte_length: int) -> WaveformPack:
        statement = select(WaveformPack).where(WaveformPack.sealed.is_(False)).order_by(WaveformPack.size.desc()).with_for_update()
        for pack in self._session.execute(statement).scalars():
            if pack.size + byte_length <= self._config.target_pack_bytes:
                return pack
            pack.sealed = True
        return self._create_pack(size=0, used_bytes=0, sealed=False)

    def _create_pack(self, size: int, used_bytes: int, sealed: bool) -> WaveformPack:
        pack = WaveformPack(path=f"{self._config.path_prefix}/{uuid.uuid4()}.bin", size=size, used_bytes=used_bytes, sealed=sealed)
        self._session.add(pack)
        self._session.flush()
        self._packs[pack.id] = pack
        return pack

    def _active_pack_data(self, pack: WaveformPack) -> bytearray:
        self._packs[pack.id] = pack
        if pack.id not in self._pack_data:
            self._pack_data[pack.id] = bytearray()
            if pack.size > 0:
                self._pack_data[pack.id].extend(self._store.download(pack.path))
        return self._pack_data[pack.id]
