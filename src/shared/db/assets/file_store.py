from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from shared.db.assets.models import Checkpoint, ExtraFile
from shared.storage import ObjectStore, ObjectStoreConfig, S3ObjectStore


@dataclass(frozen=True)
class StoredBytes:
    data: bytes
    size: int
    content_hash: str


@dataclass(frozen=True)
class StoredPath:
    path: Path
    size: int
    content_hash: str


def object_store(config: ObjectStoreConfig | None = None) -> ObjectStore:
    return S3ObjectStore(config or ObjectStoreConfig.from_env())


def checkpoint_tar(folder_path: Path) -> StoredBytes:
    assert folder_path.is_dir(), f"checkpoint path must be a folder: {folder_path}"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for file_path in _folder_files(folder_path):
            relative_path = file_path.relative_to(folder_path).as_posix()
            info = archive.gettarinfo(str(file_path), arcname=relative_path)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with file_path.open("rb") as file:
                archive.addfile(info, file)
    return stored_bytes(buffer.getvalue())


def stored_bytes(data: bytes) -> StoredBytes:
    return StoredBytes(
        data=data, size=len(data), content_hash=hashlib.sha256(data).hexdigest()
    )


def stored_path(path: Path) -> StoredPath:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return StoredPath(path=path, size=size, content_hash=digest.hexdigest())


def checkpoint_cache_path(item: Checkpoint, store: ObjectStore | None = None) -> Path:
    target = _cache_root() / "checkpoints" / item.content_hash
    if target.exists():
        return target
    data = (store or object_store()).download(item.path)
    _assert_content(item.size, item.content_hash, data)
    temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=_cache_parent(target)))
    try:
        _extract_tar(data, temp_dir)
        if not target.exists():
            temp_dir.rename(target)
        else:
            shutil.rmtree(temp_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return target


def extra_file_cache_path(item: ExtraFile, store: ObjectStore | None = None) -> Path:
    target = _cache_root() / "extra-files" / item.content_hash / item.name
    if target.exists():
        return target
    data = (store or object_store()).download(item.path)
    _assert_content(item.size, item.content_hash, data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def _folder_files(folder_path: Path) -> Iterable[Path]:
    return sorted(path for path in folder_path.rglob("*") if path.is_file())


def _cache_root() -> Path:
    return Path(os.environ.get("RUNFLOW_ASSET_CACHE_DIR", ".cache/runflow/assets"))


def _cache_parent(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.parent


def _assert_content(size: int, content_hash: str, data: bytes) -> None:
    assert len(data) == size, "cached asset size mismatch"
    assert hashlib.sha256(data).hexdigest() == content_hash, (
        "cached asset hash mismatch"
    )


def _extract_tar(data: bytes, target: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            _assert_safe_member(member)
            output_path = target / member.name
            if member.isdir():
                output_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                assert extracted is not None, f"missing tar member data: {member.name}"
                output_path.write_bytes(extracted.read())
            else:
                raise ValueError(f"unsupported checkpoint tar member: {member.name}")


def _assert_safe_member(member: tarfile.TarInfo) -> None:
    path = Path(member.name)
    assert not path.is_absolute(), f"absolute checkpoint tar member: {member.name}"
    assert ".." not in path.parts, f"unsafe checkpoint tar member: {member.name}"
