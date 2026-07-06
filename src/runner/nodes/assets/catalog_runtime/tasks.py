from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.assets.catalog_runtime.persistence import checkpoint_payload, ensure_checkpoint_bundle, ensure_extra_file, extra_file_payload
from runner.nodes.assets.catalog_runtime.specs import official_styletts_specs, papercup_plbert_spec, styletts2_utils_specs, vokan_styletts_spec
from runner.nodes.assets.catalog_runtime.types import CatalogTask
from runner.nodes.assets.catalog_runtime.validation import styletts_checkpoint_metadata


def run_catalog_task(key: str) -> dict[str, Any]:
    if key not in CATALOG_DOWNLOAD_TASKS:
        raise ValueError("styletts2_download_task_unknown")
    return CATALOG_DOWNLOAD_TASKS[key].run()


def bootstrap_styletts2_utils_assets() -> dict[str, Any]:
    asr_spec, f0_spec, plbert_spec, ood_spec = styletts2_utils_specs()
    asr, asr_skip = ensure_checkpoint_bundle(asr_spec)
    f0, f0_skip = ensure_checkpoint_bundle(f0_spec)
    plbert, plbert_skip = ensure_checkpoint_bundle(plbert_spec)
    ood, ood_skip = ensure_extra_file(ood_spec)
    return {
        "asr_checkpoint": checkpoint_payload(asr, skipped=asr_skip),
        "f0_checkpoint": checkpoint_payload(f0, skipped=f0_skip),
        "plbert_checkpoint": checkpoint_payload(plbert, skipped=plbert_skip),
        "ood_text_set": extra_file_payload(ood, skipped=ood_skip),
    }


def bootstrap_official_styletts2_checkpoints() -> dict[str, Any]:
    checkpoints = []
    for spec in official_styletts_specs():
        item, skipped = ensure_checkpoint_bundle(spec)
        checkpoints.append(checkpoint_payload(item, skipped=skipped, filename=spec.files[0].name))
    return {"checkpoints": checkpoints}


def bootstrap_papercup_multilingual_pl_bert() -> dict[str, Any]:
    item, skipped = ensure_checkpoint_bundle(papercup_plbert_spec())
    return {"plbert_checkpoint": checkpoint_payload(item, skipped=skipped)}


def bootstrap_vokan_styletts2_checkpoint() -> dict[str, Any]:
    spec = vokan_styletts_spec()
    item, skipped = ensure_checkpoint_bundle(spec)
    return {"checkpoints": [checkpoint_payload(item, skipped=skipped, filename=spec.files[0].name)]}


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
