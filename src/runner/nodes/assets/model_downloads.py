from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from uuid import UUID

import whisper

from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.assets.credentials import huggingface_token
from runner.nodes.models import CheckpointRef
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointCreate
from shared.logging_setup import get_logger


logger = get_logger(__name__)


def single_checkpoint_file(root: Path, suffixes: tuple[str, ...]) -> Path:
    """Return the single weight file with one of ``suffixes`` inside an extracted checkpoint folder."""
    wanted = tuple(suffix.lower() for suffix in suffixes)
    candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in wanted)
    if not candidates:
        raise RuntimeError(f"checkpoint_missing_file:{','.join(wanted)}:{root}")
    return candidates[0]


def ensure_model_checkpoint(kind: str, model_id: str, download: Callable[[Path], None]) -> CheckpointRef:
    """Download ``model_id`` (if not already registered) and return its resolved CheckpointRef.

    ``kind`` is the checkpoint ``type_`` (e.g. "whisper", "parakeet"). Idempotent: a checkpoint whose
    metadata records the same ``kind``/``model_id`` is reused instead of re-downloaded. ``download`` is
    called with a fresh empty folder and must populate it with the model's weight file(s). The
    download's tqdm/prints land in the node log automatically while the node is executing.
    """
    existing_id = _find_model_checkpoint_id(kind, model_id)
    if existing_id is not None:
        ref = resolve_checkpoint_ref(str(existing_id), kind)
        logger.info("using cached %s checkpoint for %s", kind, model_id)
        return ref
    with TemporaryDirectory(prefix=f"runflow-model-{kind}-") as tmp:
        folder = Path(tmp)
        logger.info("downloading %s model %s", kind, model_id)
        download(folder)
        if not any(path.is_file() for path in folder.rglob("*")):
            raise RuntimeError(f"model_download_empty:{kind}:{model_id}")
        logger.info("registering %s checkpoint for %s", kind, model_id)
        with database_session() as session:
            created = asset_crud.create_checkpoint(
                session,
                CheckpointCreate(
                    name=f"{kind}:{model_id}",
                    folder_path=folder,
                    type_=kind,
                    metadata={"model_kind": kind, "model_id": model_id, "source": "download"},
                    job_id=None,
                ),
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
