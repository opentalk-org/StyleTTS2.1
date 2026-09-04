from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from runner.nodes.assets.checkpoints import is_scratch_checkpoint
from runner.nodes.models import AssetBundleRef, CheckpointRef
from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session
from traintts.build_config import PLBERT_YAML, load_yaml
from traintts.main import RunSpec


def build_configs(
    *,
    dataset_id: str,
    phoneme_alphabet: list[str],
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    settings: Any,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    assets, asset_metadata = _assets(pretrained_assets)
    scratch = is_scratch_checkpoint(base_checkpoint)
    base_name = None
    if not scratch:
        base_name = "base_checkpoint"
        checkpoint = asset_crud.get_checkpoint(base_checkpoint.checkpoint_id)
        assets[base_name] = {"object": checkpoint.path, "entrypoint": None}

    spec = RunSpec(
        dataset_id=dataset_id,
        output_dir=str(output_dir),
        run_name=settings.display_name,
        base_checkpoint=base_name,
        asr_model=_asset_name(assets, "asr_bundle"),
        f0_model=_asset_name(assets, "f0_model"),
        plbert=_asset_name(assets, "plbert"),
        plbert_symbols=asset_metadata["plbert"]["phoneme_symbols"],
        plbert_languages=asset_metadata["plbert"]["languages"],
        seed=settings.seed,
        validation_samples=settings.validation_samples,
        learning_rate=settings.learning_rate,
        precision=settings.numeric_precision.value,
        validation_every_steps=settings.validation_interval_steps,
        checkpoint_every_steps=settings.checkpoint_interval_steps,
        log_every_steps=settings.log_interval_steps,
        profiling_enabled=settings.profiling_enabled,
        distributed_processes=settings.distributed_processes,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        load_optimizer=settings.load_optimizer,
        reset_training_step=settings.reset_training_step,
        decoder=settings.decoder.value,
        multispeaker=settings.multispeaker,
        checkpoint_decoder_gradients=settings.checkpoint_decoder_gradients,
        checkpoint_discriminator_gradients=settings.checkpoint_discriminator_gradients,
        symbols=phoneme_alphabet,
        training_stages=settings.training_stages,
    )
    accumulation = settings.gradient_accumulation_steps
    sequences = [
        {
            "batches": stage.steps * accumulation,
            "max_seconds": stage.max_audio_seconds,
        }
        for stage in settings.training_stages
    ]
    plbert = load_yaml(PLBERT_YAML)
    data_config = {
        "dataset_id": dataset_id,
        "seed": settings.seed,
        "max_text_tokens": plbert["model_params"]["max_position_embeddings"],
        "plbert_languages": spec.plbert_languages,
        "assets": assets,
        "validation": {
            "samples": settings.validation_samples,
            "max_seconds": max(sequence["max_seconds"] for sequence in sequences),
        },
        "training": sequences,
    }
    train_config = yaml.safe_dump(
        spec.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    return data_config, train_config


def _asset_name(assets: dict[str, dict[str, Any]], role: str) -> str | None:
    return role if role in assets else None


def _assets(
    ref: AssetBundleRef | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    assets: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {"plbert": {
        "phoneme_symbols": [],
        "languages": [],
    }}
    if ref is None:
        return assets, metadata
    with database_session() as session:
        for item in ref.metadata["assets"]:
            role = str(item["role"])
            asset_id = UUID(str(item["id"]))
            if item["storage"] == "checkpoint":
                record = asset_crud.get_checkpoint(asset_id)
                root = asset_crud.get_checkpoint_path(session, asset_id)
                entrypoint = Path(str(item["path"])).relative_to(root).as_posix()
            else:
                record = asset_crud.get_extra_file(asset_id)
                entrypoint = None
            assets[role] = {
                "object": record.path,
                "entrypoint": entrypoint,
            }
            if role == "plbert":
                metadata[role] = dict(item["metadata"])
    return assets, metadata
