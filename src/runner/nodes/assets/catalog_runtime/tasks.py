from __future__ import annotations

import logging
from typing import Any

from runner.nodes.assets.catalog_runtime.entries import CATALOG_ENTRIES, CatalogKey
from runner.nodes.assets.catalog_runtime.persistence import checkpoint_payload, ensure_checkpoint_bundle, ensure_extra_file, extra_file_payload
from runner.nodes.assets.catalog_runtime.specs import official_styletts_specs, papercup_plbert_spec, styletts2_utils_specs, vokan_styletts_spec
from runner.nodes.assets.catalog_runtime.types import CatalogTask
from runner.nodes.assets.model_downloads import (
    download_hf_snapshot,
    download_nemo_snapshot,
    download_raon_model_files,
    download_whisper_model_files,
    ensure_model_checkpoint,
)


_NEMO_ASR_KINDS = {"parakeet", "canary", "sortformer"}
_MOS_BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
_TTS_DEFAULT_REPOS = {
    entry.item: entry.file
    for entry in CATALOG_ENTRIES
    if entry.catalog_key is CatalogKey.TTS_MODELS
}
_SMART_TURN_ENTRY = next(entry for entry in CATALOG_ENTRIES if entry.catalog_key is CatalogKey.TURN_MODELS)
_SMART_TURN_MODEL = _SMART_TURN_ENTRY.item
_SMART_TURN_FILE = _SMART_TURN_ENTRY.file

# Per-engine snapshot filters so we only download the files each engine's ``load()`` actually reads.
# Repos ship the same weights in several redundant formats / checkpoint variants; without a filter
# the whole repo is pulled and most of it never loaded. Keys map to snapshot_download kwargs.
_TTS_DOWNLOAD_FILTERS: dict[str, dict[str, list[str]]] = {
    # transformers loads the sharded ``.safetensors``; the ``.pth`` + ``.bin`` are the same weights again.
    "dia": {"ignore_patterns": ["dia-v1.pth", "pytorch_model.bin"]},
    # chatterbox.from_local reads a fixed file set (multilingual + english modes); everything else in the
    # repo is alternate variants (t3_mtl23ls_v3, t3_23lang, s3gen_v3, *.pt duplicates) we never load.
    "chatterbox": {
        "allow_patterns": [
            "ve.pt",
            "ve.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "t3_cfg.safetensors",
            "s3gen.pt",
            "s3gen.safetensors",
            "tokenizer.json",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
            "conds.pt",
        ]
    },
    # F5 loader uses only the F5TTS_v1_Base variant (weights + vocab); the repo also ships Base / bigvgan
    # / no_zero_init variants, each duplicated as ``.pt`` and ``.safetensors``.
    "f5_tts": {"allow_patterns": ["F5TTS_v1_Base/*"]},
}

# whisperx alignment loads a transformers Wav2Vec2 model: weights + config + vocab only. Skip the Flax
# ``.msgpack`` copy, the KenLM ``language_model/`` (unused by alignment), and the repo's eval artifacts.
_WHISPERX_IGNORE_PATTERNS = ["*.msgpack", "language_model/*", "log_*.txt", "*eval_results*.txt", "eval.py", "full_eval.sh"]


_LOGGER = logging.getLogger(__name__)


def run_catalog_task(key: str, item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    if key not in CATALOG_DOWNLOAD_TASKS:
        raise ValueError("styletts2_download_task_unknown")
    log.info("catalog task starting key=%s item=%s", key, item or "<all>")
    result = CATALOG_DOWNLOAD_TASKS[key].run(item, logger=log)
    log.info("catalog task finished key=%s item=%s", key, item or "<all>")
    return result


def bootstrap_styletts2_utils_assets(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    specs = styletts2_utils_specs()
    keys = {
        "styletts2_utils_asr": "asr_checkpoint",
        "styletts2_utils_f0": "f0_checkpoint",
        "styletts2_utils_plbert": "plbert_checkpoint",
        "styletts2_libritts_ood": "ood_text_set",
    }
    selected = _selected_specs(specs, item)
    payload: dict[str, Any] = {}
    for spec in selected:
        output_key = keys[spec.key]
        if spec.key == "styletts2_libritts_ood":
            extra, skipped = ensure_extra_file(spec, logger=log)
            payload[output_key] = extra_file_payload(extra, skipped=skipped)
        else:
            checkpoint, skipped = ensure_checkpoint_bundle(spec, logger=log)
            payload[output_key] = checkpoint_payload(checkpoint, skipped=skipped)
    return payload


def bootstrap_official_styletts2_checkpoints(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    checkpoints = []
    for spec in _selected_specs(official_styletts_specs(), item):
        item, skipped = ensure_checkpoint_bundle(spec, logger=log)
        checkpoints.append(checkpoint_payload(item, skipped=skipped, filename=spec.files[0].name))
    return {"checkpoints": checkpoints}


def bootstrap_papercup_multilingual_pl_bert(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    spec = papercup_plbert_spec()
    _assert_single_item(spec.key, item)
    checkpoint, skipped = ensure_checkpoint_bundle(spec, logger=log)
    return {"plbert_checkpoint": checkpoint_payload(checkpoint, skipped=skipped)}


def bootstrap_vokan_styletts2_checkpoint(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    spec = vokan_styletts_spec()
    _assert_single_item(spec.key, item)
    checkpoint, skipped = ensure_checkpoint_bundle(spec, logger=log)
    return {"checkpoints": [checkpoint_payload(checkpoint, skipped=skipped, filename=spec.files[0].name)]}


def bootstrap_asr_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    kind, model_id = _parse_asr_item(item)
    if kind == "whisper":
        download = lambda folder: download_whisper_model_files(model_id, folder)
    elif kind in _NEMO_ASR_KINDS:
        download = lambda folder: download_nemo_snapshot(model_id, folder)
    elif kind == "whisperx":
        download = lambda folder: download_hf_snapshot(model_id, folder, ignore_patterns=_WHISPERX_IGNORE_PATTERNS)
    else:
        raise ValueError(f"catalog_item_unknown:{item}")
    log.info("asr model download starting kind=%s model=%s", kind, model_id)
    ref = ensure_model_checkpoint(kind, model_id, download)
    log.info("asr model download resolved kind=%s model=%s checkpoint=%s", kind, model_id, ref.checkpoint_id)
    return {
        "model_checkpoint": {
            "kind": kind,
            "model_id": model_id,
            "checkpoint_id": str(ref.checkpoint_id),
            "name": ref.name,
        }
    }


def bootstrap_tts_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    engine, repo = _parse_tts_item(item)
    log.info("tts model download starting engine=%s repo=%s", engine, repo)
    filters = _TTS_DOWNLOAD_FILTERS.get(engine, {})
    if engine == "raon_opentts":
        download = lambda folder: download_raon_model_files(repo, folder)
        validate = _raon_checkpoint_valid
    else:
        download = lambda folder: download_hf_snapshot(repo, folder, **filters)
        validate = None
    ref = ensure_model_checkpoint(engine, repo, download, validate)
    log.info("tts model download resolved engine=%s repo=%s checkpoint=%s", engine, repo, ref.checkpoint_id)
    return {
        "model_checkpoint": {
            "engine": engine,
            "repo": repo,
            "checkpoint_id": str(ref.checkpoint_id),
            "name": ref.name,
        }
    }


def _raon_checkpoint_valid(path: Path) -> bool:
    return (
        (path / "runtime" / "raon_f5_tts" / "infer" / "utils_infer.py").is_file()
        and (path / "vocoder" / "generator.ckpt").is_file()
        and (path / "config.yaml").is_file()
        and (path / "vocab.txt").is_file()
    )


def bootstrap_mos_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    model_id = item.strip()
    if model_id != _MOS_BASE_MODEL:
        raise ValueError(f"catalog_item_unknown:{item}")
    log.info("MOS base model download starting model=%s", model_id)
    ref = ensure_model_checkpoint(
        "mos_base",
        model_id,
        lambda folder: download_hf_snapshot(
            model_id,
            folder,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        ),
    )
    log.info("MOS base model download resolved model=%s checkpoint=%s", model_id, ref.checkpoint_id)
    return {
        "model_checkpoint": {
            "kind": "mos_base",
            "model_id": model_id,
            "checkpoint_id": str(ref.checkpoint_id),
            "name": ref.name,
        }
    }


def bootstrap_turn_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    model_id = item.strip()
    if model_id != _SMART_TURN_MODEL:
        raise ValueError(f"catalog_item_unknown:{item}")
    log.info("Smart Turn model download starting model=%s", model_id)
    ref = ensure_model_checkpoint(
        "smart_turn",
        model_id,
        lambda folder: download_hf_snapshot(model_id, folder, allow_patterns=[_SMART_TURN_FILE]),
    )
    log.info("Smart Turn model download resolved model=%s checkpoint=%s", model_id, ref.checkpoint_id)
    return {
        "model_checkpoint": {
            "kind": "smart_turn",
            "model_id": model_id,
            "checkpoint_id": str(ref.checkpoint_id),
            "name": ref.name,
        }
    }


def _parse_tts_item(item: str) -> tuple[str, str]:
    requested = item.strip()
    engine, separator, repo = requested.partition(":")
    engine = engine.strip()
    if engine not in _TTS_DEFAULT_REPOS:
        raise ValueError(f"catalog_item_unknown:{item}")
    repo_id = repo.strip() if separator and repo.strip() else _TTS_DEFAULT_REPOS[engine]
    return engine, repo_id


def _parse_asr_item(item: str) -> tuple[str, str]:
    requested = item.strip()
    kind, separator, model_id = requested.partition(":")
    if not separator or not kind or not model_id:
        raise ValueError(f"catalog_item_unknown:{item}")
    return kind.strip(), model_id.strip()


def _assert_single_item(spec_key: str, item: str) -> None:
    requested = item.strip()
    if requested and requested != spec_key:
        raise ValueError(f"catalog_item_unknown:{requested}")


def _selected_specs(specs: tuple[Any, ...], item: str) -> tuple[Any, ...]:
    requested = item.strip()
    if not requested:
        return specs
    selected = tuple(spec for spec in specs if spec.key == requested)
    if not selected:
        raise ValueError(f"catalog_item_unknown:{requested}")
    return selected


CATALOG_DOWNLOAD_TASKS: dict[str, CatalogTask] = {
    "styletts2_utils": CatalogTask(
        key="styletts2_utils",
        run=bootstrap_styletts2_utils_assets,
    ),
    "official_checkpoints": CatalogTask(
        key="official_checkpoints",
        run=bootstrap_official_styletts2_checkpoints,
    ),
    "papercup_multilingual_pl_bert": CatalogTask(
        key="papercup_multilingual_pl_bert",
        run=bootstrap_papercup_multilingual_pl_bert,
    ),
    "vokan_checkpoint": CatalogTask(
        key="vokan_checkpoint",
        run=bootstrap_vokan_styletts2_checkpoint,
    ),
    "asr_models": CatalogTask(
        key="asr_models",
        run=bootstrap_asr_model,
    ),
    "tts_models": CatalogTask(
        key="tts_models",
        run=bootstrap_tts_model,
    ),
    "mos_models": CatalogTask(
        key="mos_models",
        run=bootstrap_mos_model,
    ),
    "turn_models": CatalogTask(
        key="turn_models",
        run=bootstrap_turn_model,
    ),
}
