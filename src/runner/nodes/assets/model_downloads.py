from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from urllib.request import urlopen
from uuid import UUID

import whisper

from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.assets.credentials import huggingface_token
from runner.nodes.models import CheckpointRef
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointCreate, CheckpointUpdate
from shared.logging_setup import get_logger


logger = get_logger(__name__)


def single_checkpoint_file(root: Path, suffixes: tuple[str, ...]) -> Path:
    """Return the single weight file with one of ``suffixes`` inside an extracted checkpoint folder."""
    wanted = tuple(suffix.lower() for suffix in suffixes)
    candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in wanted)
    if not candidates:
        raise RuntimeError(f"checkpoint_missing_file:{','.join(wanted)}:{root}")
    return candidates[0]


def ensure_model_checkpoint(
    kind: str,
    model_id: str,
    download: Callable[[Path], None],
    validate: Callable[[Path], bool] | None = None,
) -> CheckpointRef:
    """Download ``model_id`` (if not already registered) and return its resolved CheckpointRef.

    ``kind`` is the checkpoint ``type_`` (e.g. "whisper", "parakeet"). Idempotent: a checkpoint whose
    metadata records the same ``kind``/``model_id`` is reused instead of re-downloaded. ``download`` is
    called with a fresh empty folder and must populate it with the model's weight file(s). The
    download's tqdm/prints land in the node log automatically while the node is executing.
    """
    existing_id = _find_model_checkpoint_id(kind, model_id)
    existing_path: Path | None = None
    if existing_id is not None:
        try:
            ref = resolve_checkpoint_ref(str(existing_id), kind)
            if validate is None or validate(ref.path):
                logger.info("using cached %s checkpoint for %s", kind, model_id)
                return ref
            logger.info("cached %s checkpoint for %s is incomplete; re-downloading", kind, model_id)
            existing_path = ref.path
        except Exception:
            logger.info("cached %s checkpoint for %s unavailable; re-downloading", kind, model_id)
    with TemporaryDirectory(prefix=f"runflow-model-{kind}-") as tmp:
        folder = Path(tmp)
        if existing_path is not None:
            shutil.copytree(
                existing_path,
                folder,
                dirs_exist_ok=True,
                copy_function=os.link,
            )
        logger.info("downloading %s model %s", kind, model_id)
        download(folder)
        if not any(path.is_file() for path in folder.rglob("*")):
            raise RuntimeError(f"model_download_empty:{kind}:{model_id}")
        logger.info("registering %s checkpoint for %s", kind, model_id)
        with database_session() as session:
            payload = CheckpointUpdate(
                name=f"{kind}:{model_id}",
                folder_path=folder,
                type_=kind,
                metadata={"model_kind": kind, "model_id": model_id, "source": "download"},
                job_id=None,
            )
            if existing_id is None:
                created = asset_crud.create_checkpoint(
                    session,
                    CheckpointCreate(
                        name=payload.name,
                        folder_path=folder,
                        type_=payload.type_,
                        metadata=payload.metadata,
                        job_id=payload.job_id,
                    ),
                )
            else:
                created = asset_crud.update_checkpoint(
                    session,
                    existing_id,
                    payload,
                )
            checkpoint_id = created.id
    return resolve_checkpoint_ref(str(checkpoint_id), kind)


def _find_model_checkpoint_id(kind: str, model_id: str) -> UUID | None:
    with database_session() as session:
        for checkpoint in asset_crud.list_checkpoints(session):
            metadata = checkpoint.metadata_ or {}
            if checkpoint.type_ == kind and metadata.get("model_id") == model_id:
                return checkpoint.id
    return None


def download_whisper_model_files(version: str, folder: Path) -> None:
    """Download a Whisper checkpoint (single ``.pt``) into ``folder`` without loading it into memory."""
    whisper._download(whisper._MODELS[version], str(folder), False)


def _disable_hf_progress_bars() -> None:
    """Turn off huggingface_hub's threaded tqdm progress bars.

    snapshot_download parallelises file fetches with tqdm.contrib.concurrent.thread_map.
    Inside the long-running runner (with stdout teed into per-node logs) that tqdm can
    crash on teardown with ``type object 'tqdm' has no attribute '_lock'`` — the class-level
    ``_lock`` gets lost across the process's many tqdm users. Disabling the bars removes the
    threaded tqdm entirely; download progress is already logged at start/finish.
    """
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()


def download_hf_snapshot(
    model_id: str,
    folder: Path,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    """Download a HuggingFace model snapshot (e.g. a wav2vec2 aligner) into ``folder``.

    ``allow_patterns``/``ignore_patterns`` are forwarded to ``snapshot_download`` so callers only
    fetch the files their loader actually reads. Many TTS/ASR repos ship the same weights in several
    redundant formats (``.pth`` + ``.bin`` + ``.safetensors``, Flax ``.msgpack``, extra checkpoint
    variants); without a filter the whole repo is pulled and most of it is never loaded.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub_not_installed") from exc
    _disable_hf_progress_bars()
    snapshot_download(
        repo_id=model_id,
        local_dir=str(folder),
        token=huggingface_token(),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def download_nemo_snapshot(model_id: str, folder: Path) -> None:
    """Download a NeMo model's ``.nemo`` file(s) from HuggingFace into ``folder``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub_not_installed") from exc
    _disable_hf_progress_bars()
    with TemporaryDirectory(prefix="runflow-hf-") as raw:
        raw_dir = Path(raw)
        snapshot_download(repo_id=model_id, local_dir=str(raw_dir), allow_patterns=["*.nemo"], token=huggingface_token())
        nemo_files = sorted(raw_dir.rglob("*.nemo"))
        if not nemo_files:
            raise RuntimeError(f"nemo_checkpoint_not_found_in_repo:{model_id}")
        for source in nemo_files:
            shutil.copy2(source, folder / source.name)


RAON_RUNTIME_COMMIT = "0dd28405c9348e9505c2a1c92a250cb3beffd950"


def download_raon_model_files(model_id: str, folder: Path) -> None:
    download_hf_snapshot(model_id, folder)
    with TemporaryDirectory(prefix="runflow-raon-source-") as tmp:
        archive_path = Path(tmp) / "source.tar.gz"
        _download_file(
            f"https://github.com/krafton-ai/Raon-OpenTTS/archive/{RAON_RUNTIME_COMMIT}.tar.gz",
            archive_path,
        )
        _extract_raon_runtime(archive_path, folder / "runtime" / "raon_f5_tts")
    vocoder_dir = folder / "vocoder"
    vocoder_dir.mkdir()
    _download_file(
        "https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz/resolve/main/generator.ckpt",
        vocoder_dir / "generator.ckpt",
    )


def _download_file(url: str, target: Path) -> None:
    with urlopen(url, timeout=120) as response, target.open("wb") as destination:
        shutil.copyfileobj(response, destination)


def _extract_raon_runtime(archive_path: Path, target: Path) -> None:
    wanted = (
        "infer/",
        "model/",
    )
    target.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            marker = "/src/f5_tts/"
            if marker not in member.name or not member.isfile():
                continue
            relative = member.name.split(marker, maxsplit=1)[1]
            if not relative.startswith(wanted):
                continue
            relative_path = Path(relative)
            assert not relative_path.is_absolute() and ".." not in relative_path.parts, (
                f"unsafe Raon runtime archive member: {member.name}"
            )
            output = target / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None, f"missing Raon runtime archive member: {member.name}"
            with source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    (target / "__init__.py").write_text("", encoding="utf-8")
    (target / "model").mkdir(exist_ok=True)
    (target / "model" / "__init__.py").write_text(
        "from .backbones.dit import DiT\nfrom .cfm import CFM\n",
        encoding="utf-8",
    )
