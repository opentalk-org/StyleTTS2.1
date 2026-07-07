from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.models import AssetBundleRef, CheckpointRef, TrainingManifest
from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS
from runner.nodes.training.styletts.finetune import config as styletts_config
from runner.nodes.training.styletts.finetune import layout as styletts_layout


def resolve_symbol_list(alphabet: list[str] | None) -> list[str]:
    """Return the symbol table used to index the text embedding.

    StyleTTS2 (LJSpeech / LibriTTS / Vokan) is trained with the canonical
    178-entry single-character IPA symbol set, and the pretrained text-encoder /
    text-aligner embeddings have exactly that many rows. A finetune must reuse
    that table so the pretrained embeddings stay aligned. We only honour a
    provided alphabet if it is a genuine single-character table of the same size;
    anything else (e.g. the legacy multi-character token list) falls back to the
    canonical set instead of silently resizing embeddings."""
    canonical = [str(symbol) for symbol in DEFAULT_STYLETTS_SYMBOLS]
    if not alphabet:
        return canonical
    symbols = [str(symbol) for symbol in alphabet]
    if len(symbols) == len(canonical) and all(len(symbol) == 1 for symbol in symbols):
        return symbols
    return canonical


def build_node_config(
    *,
    manifest: TrainingManifest,
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    ood_text_sets: dict[str, Any],
    settings: Any,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    config_path = output_dir / "config.yaml"
    symbol_list = resolve_symbol_list(manifest.phoneme_alphabet)
    symbol_count = len(symbol_list)
    symbols = symbol_list
    base_root = base_checkpoint.path
    pretrained_model = styletts_layout.latest_weight(base_root)
    asset_paths = _training_asset_paths(pretrained_assets)
    styletts_yaml = styletts_config.build_config(
        log_dir=output_dir / "run",
        train_list=str(manifest.metadata["train_manifest_path"]),
        validation_list=str(manifest.metadata["validation_manifest_path"]),
        root_path=str(manifest.metadata["root_path"]),
        ood_texts=str(_ood_text_path(ood_text_sets, asset_paths, output_dir)),
        pretrained_model=pretrained_model,
        asr_config=_asr_config(symbol_count),
        asr_path=asset_paths["asr_bundle"],
        f0_path=asset_paths["f0_model"],
        plbert_config=_plbert_config(symbol_count),
        plbert_path=asset_paths["plbert"],
        epochs=settings.epochs_base + settings.epochs_diffusion + settings.epochs_joint,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        max_len=int(settings.max_sequence_seconds * 30),
        diff_epoch=settings.epochs_base,
        joint_epoch=settings.epochs_base + settings.epochs_diffusion,
        save_every_n_epochs=settings.save_interval_epochs,
        load_optimizer=settings.load_optimizer,
        generator_checkpointing=settings.checkpoint_each_stage,
        discriminators_checkpointing=settings.checkpoint_each_stage,
        precision=settings.numeric_precision.value,
        slmadv_min_len=settings.slmadv_min_len,
        slmadv_max_len=settings.slmadv_max_len,
        slmadv_batch_samples=settings.slmadv_batch_samples,
        slmadv_scale=settings.slm_scale,
        architecture_path=styletts_layout.architecture_yaml(base_root),
        multispeaker=settings.multispeaker,
        decoder_type=settings.decoder.value,
        studio_publish={
            "enabled": True,
            "parent_checkpoint_id": str(base_checkpoint.checkpoint_id),
            "parent_checkpoint_path": str(base_checkpoint.path),
            "base_library_root": str(base_checkpoint.path),
            "pretrained_relpath": _relative_weight(base_root, pretrained_model),
            "run_name": settings.display_name,
        },
        symbols=symbols,
        symbol_count=symbol_count,
    )
    styletts_config.write_config(config_path, styletts_yaml)
    return config_path, styletts_yaml


def _training_asset_paths(ref: AssetBundleRef | None) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {"asr_bundle": None, "f0_model": None, "plbert": None, "ood_text_set": None}
    if ref is None:
        return paths
    assets = ref.metadata["assets"]
    for asset in assets:
        role = str(asset["role"])
        if role in paths and paths[role] is None:
            paths[role] = Path(str(asset["path"]))
    return paths


def _ood_text_path(ood_text_sets: dict[str, Any], asset_paths: dict[str, Path | None], output_dir: Path) -> Path:
    if "path" in ood_text_sets:
        path = Path(str(ood_text_sets["path"]))
        if path.is_file():
            return path
    if "paths" in ood_text_sets:
        paths = [Path(str(item)) for item in ood_text_sets["paths"]]
        if len(paths) == 1 and paths[0].is_file():
            return paths[0]
        combined = output_dir / "ood_texts.txt"
        combined.write_text("\n".join(path.read_text(encoding="utf-8").strip() for path in paths), encoding="utf-8")
        return combined
    asset_path = asset_paths["ood_text_set"]
    if asset_path is not None and asset_path.is_file():
        return asset_path
    raise ValueError("StyleTTS finetune config requires an OOD text file path")


def _asr_config(symbol_count: int) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.ASR_YAML)
    config["model_params"]["n_token"] = int(symbol_count)
    return config


def _plbert_config(symbol_count: int) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.PLBERT_YAML)
    config["model_params"]["vocab_size"] = int(symbol_count)
    return config


def _relative_weight(root: Path, weight: Path) -> str:
    try:
        return weight.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return weight.name
