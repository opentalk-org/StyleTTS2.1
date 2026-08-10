from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors import safe_open

from runner.nodes.assets.checkpoints import is_scratch_checkpoint
from runner.nodes.models import AssetBundleRef, CheckpointRef
from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS
from runner.nodes.training.styletts.finetune import config as styletts_config
from runner.nodes.training.styletts.finetune import layout as styletts_layout


def resolve_symbol_list(alphabet: list[str] | None) -> list[str]:
    """Keep pretrained rows stable and allow explicitly appended tokens."""
    canonical = [str(symbol) for symbol in DEFAULT_STYLETTS_SYMBOLS]
    if not alphabet:
        return canonical
    symbols = [str(symbol) for symbol in alphabet]
    if symbols[:len(canonical)] != canonical:
        raise ValueError(
            "StyleTTS finetune alphabet must retain the canonical symbol prefix; "
            "append custom tokens after <end/>"
        )
    appended = symbols[len(canonical):]
    if len(set(appended)) != len(appended) or any(
        symbol in canonical for symbol in appended
    ):
        raise ValueError("StyleTTS finetune appended alphabet contains duplicate tokens")
    return symbols


def build_node_config(
    *,
    dataset_id: str,
    validation_samples: int,
    phoneme_alphabet: list[str],
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    settings: Any,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    config_path = output_dir / "config.yaml"
    symbol_list = resolve_symbol_list(phoneme_alphabet)
    symbol_count = len(symbol_list)
    symbols = symbol_list
    scratch = is_scratch_checkpoint(base_checkpoint)
    asset_paths, asset_metadata = _training_asset_paths(pretrained_assets)
    if scratch:
        _require_scratch_assets(asset_paths)
        pretrained_model = None
        architecture_path = None
    else:
        base_root = base_checkpoint.path
        pretrained_model = styletts_layout.latest_weight(base_root)
        architecture_path = styletts_layout.architecture_yaml(base_root)
    styletts_yaml = styletts_config.build_config(
        log_dir=output_dir / "run",
        dataset_id=dataset_id,
        validation_samples=validation_samples,
        pretrained_model=pretrained_model,
        asr_config=_asr_config(symbol_count),
        asr_path=asset_paths["asr_bundle"],
        f0_path=asset_paths["f0_model"],
        plbert_config=_plbert_config(
            symbol_list,
            asset_paths["plbert"],
            asset_metadata["plbert"],
        ),
        plbert_path=asset_paths["plbert"],
        total_steps=sum(stage.steps for stage in settings.training_stages),
        seed=settings.seed,
        learning_rate=settings.learning_rate,
        training_stages=settings.training_stages,
        validation_every_steps=settings.validation_interval_steps,
        checkpoint_every_steps=settings.checkpoint_interval_steps,
        log_every_steps=settings.log_interval_steps,
        profiling_enabled=settings.profiling_enabled,
        distributed_processes=settings.distributed_processes,
        load_optimizer=settings.load_optimizer,
        reset_training_step=settings.reset_training_step,
        generator_checkpointing=settings.checkpoint_decoder_gradients,
        discriminators_checkpointing=settings.checkpoint_discriminator_gradients,
        precision=settings.numeric_precision.value,
        architecture_path=architecture_path,
        multispeaker=settings.multispeaker,
        decoder_type=settings.decoder.value,
        studio_publish=_studio_publish(
            scratch,
            base_checkpoint,
            pretrained_model,
            settings.display_name,
            settings.mlflow_run_id,
        ),
        symbols=symbols,
        symbol_count=symbol_count,
    )
    styletts_config.write_config(config_path, styletts_yaml)
    return config_path, styletts_yaml


def _require_scratch_assets(asset_paths: dict[str, Path | None]) -> None:
    """From-scratch training has no pretrained weights to fall back on, so the
    auxiliary ASR text-aligner, F0 pitch-extractor, and PL-BERT must be supplied;
    otherwise those modules would train from random init and never converge."""
    missing = [role for role in ("asr_bundle", "f0_model", "plbert") if asset_paths[role] is None]
    if missing:
        raise ValueError(f"StyleTTS from-scratch training requires pretrained assets: {', '.join(missing)}")


def _studio_publish(
    scratch: bool,
    base_checkpoint: CheckpointRef,
    pretrained_model: Path | None,
    run_name: str,
    mlflow_run_id: str,
) -> dict[str, Any]:
    if scratch:
        return {
            "enabled": True,
            "parent_checkpoint_id": "",
            "parent_checkpoint_path": "",
            "base_library_root": "",
            "pretrained_relpath": "",
            "run_name": run_name,
            "mlflow_run_id": mlflow_run_id,
        }
    return {
        "enabled": True,
        "parent_checkpoint_id": str(base_checkpoint.checkpoint_id),
        "parent_checkpoint_path": str(base_checkpoint.path),
        "base_library_root": str(base_checkpoint.path),
        "pretrained_relpath": _relative_weight(base_checkpoint.path, pretrained_model),
        "run_name": run_name,
        "mlflow_run_id": mlflow_run_id,
    }


def _training_asset_paths(
    ref: AssetBundleRef | None,
) -> tuple[dict[str, Path | None], dict[str, dict[str, Any]]]:
    paths: dict[str, Path | None] = {"asr_bundle": None, "f0_model": None, "plbert": None}
    metadata: dict[str, dict[str, Any]] = {"plbert": {}}
    if ref is None:
        return paths, metadata
    for asset in ref.metadata["assets"]:
        role = str(asset["role"])
        if role in paths and paths[role] is None:
            paths[role] = Path(str(asset["path"]))
            if role == "plbert":
                metadata["plbert"] = dict(asset["metadata"])
    return paths, metadata


def _asr_config(symbol_count: int) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.ASR_YAML)
    config["model_params"]["n_token"] = int(symbol_count)
    return config


def _plbert_config(
    symbols: list[str],
    path: Path | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.PLBERT_YAML)
    config["model_params"]["vocab_size"] = len(symbols)
    if path is not None and path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            positions = checkpoint.get_slice(
                "encoder._orig_mod.embeddings.position_embeddings.weight"
            ).get_shape()[0]
        config["model_params"]["max_position_embeddings"] = int(positions)
        config["input_symbols"] = symbols
        config["artifact_symbols"] = metadata["phoneme_symbols"]
        config["languages"] = metadata["languages"]
        config["modality_id"] = 0
    return config


def _relative_weight(root: Path, weight: Path) -> str:
    return weight.resolve().relative_to(root.resolve()).as_posix()
