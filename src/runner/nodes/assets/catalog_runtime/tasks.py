from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from runner.nodes.assets.catalog_runtime.persistence import checkpoint_payload, ensure_checkpoint_bundle, ensure_extra_file, extra_file_payload
from runner.nodes.assets.catalog_runtime.specs import official_styletts_specs, papercup_plbert_spec, styletts2_utils_specs, vokan_styletts_spec
from runner.nodes.assets.catalog_runtime.types import CatalogTask
from runner.nodes.assets.catalog_runtime.validation import styletts_checkpoint_metadata


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
}
