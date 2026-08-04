import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.pack_folders import PackFolderAllocator
from shared.db.staged_objects import register_staged_object
from shared.storage import ObjectStore

LOGGER = logging.getLogger(__name__)
UPLOAD_VERIFY_ATTEMPTS = 3


@dataclass(frozen=True)
class AudioPackConfig:
    target_pack_bytes: int = 128 * 1024 * 1024
    path_prefix: str = "audio-packs"
    folder_target_files: int = 256
    prune_used_ratio: float = 0.5
    prune_size_ratio: float = 0.2
    remote_workers: int = 9


@dataclass(frozen=True)
class PackedWrite:
    bucket_file: BucketFile
    byte_offset: int
    byte_length: int


class AudioPackWriter:
    def __init__(self, session: Session, store: ObjectStore, config: AudioPackConfig) -> None:
        self._session = session
        self._store = store
        self._config = config
        self._active_pack: BucketFile | None = None
        self._pack_data: dict[uuid.UUID, bytearray] = {}
        self._packs: dict[uuid.UUID, BucketFile] = {}
        self._dirty_pack_ids: set[uuid.UUID] = set()
        self._folders = PackFolderAllocator(
            session,
            BucketFile,
            config.path_prefix,
            config.folder_target_files,
        )

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

    def flush(self, verify: bool = False) -> None:
        pack_ids = list(self._dirty_pack_ids)
        payloads = [(self._packs[pack_id], bytes(self._pack_data[pack_id])) for pack_id in pack_ids]
        for pack, _ in payloads:
            register_staged_object(self._session, self._store, pack.path)
        with ThreadPoolExecutor(max_workers=self._config.remote_workers) as executor:
            list(executor.map(self._upload_pack, payloads))
            if verify:
                list(executor.map(self._verify_pack, payloads))
        for pack, _ in payloads:
            pack.sealed = True

    def _upload_pack(self, payload: tuple[BucketFile, bytes]) -> None:
        pack, data = payload
        self._store.upload(pack.path, data)

    def _verify_pack(self, payload: tuple[BucketFile, bytes]) -> None:
        pack, data = payload
        for attempt in range(UPLOAD_VERIFY_ATTEMPTS):
            if self._store.download(pack.path) == data:
                return
            if attempt < UPLOAD_VERIFY_ATTEMPTS - 1:
                LOGGER.warning(
                    "audio pack verification mismatch; re-uploading: %s",
                    pack.path,
                )
                self._store.upload(pack.path, data)
        raise IOError(f"uploaded audio pack failed verification: {pack.path}")

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
        return self._create_pack(size=0, used_bytes=0, sealed=False)

    def _create_pack(self, size: int, used_bytes: int, sealed: bool) -> BucketFile:
        pack_id = uuid.uuid4()
        pack = BucketFile(
            id=pack_id,
            path=self._folders.path_for(pack_id),
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
