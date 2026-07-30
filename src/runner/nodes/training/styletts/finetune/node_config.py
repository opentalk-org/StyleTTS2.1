from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.assets.checkpoints import is_scratch_checkpoint
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
    settings: Any,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    config_path = output_dir / "config.yaml"
    symbol_list = resolve_symbol_list(manifest.phoneme_alphabet)
    symbol_count = len(symbol_list)
    symbols = symbol_list
    scratch = is_scratch_checkpoint(base_checkpoint)
    asset_paths, ood_paths = _training_asset_paths(pretrained_assets)
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
        train_list=str(manifest.metadata["train_manifest_path"]),
        validation_list=str(manifest.metadata["validation_manifest_path"]),
        root_path=str(manifest.metadata["root_path"]),
        stream_from_buckets=bool(manifest.metadata.get("stream_from_buckets", False)),
        stream_plan_path=str(manifest.metadata.get("stream_plan_path", "")),
        cache_dir=str(manifest.metadata.get("cache_dir", "")),
        bucket_cache_budget_bytes=int(settings.bucket_cache_budget_gb * 1024 * 1024 * 1024),
        ood_texts=str(_ood_text_path(ood_paths, output_dir)),
        pretrained_model=pretrained_model,
        asr_config=_asr_config(symbol_count),
        asr_path=asset_paths["asr_bundle"],
        f0_path=asset_paths["f0_model"],
        plbert_config=_plbert_config(symbol_count),
        plbert_path=asset_paths["plbert"],
        total_steps=(
            settings.base_steps
            + settings.diffusion_steps
            + settings.joint_steps
        ),
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        max_len=int(settings.max_decoder_seconds * 80),
        max_audio_seconds=settings.max_audio_seconds,
        diffusion_start_step=settings.base_steps,
        joint_start_step=settings.base_steps + settings.diffusion_steps,
        validation_every_steps=settings.validation_interval_steps,
        checkpoint_every_steps=settings.checkpoint_interval_steps,
        log_every_steps=settings.log_interval_steps,
        profiling_enabled=settings.profiling_enabled,
        distributed_processes=settings.distributed_processes,
        load_optimizer=settings.load_optimizer,
        generator_checkpointing=settings.checkpoint_decoder_gradients,
        discriminators_checkpointing=settings.checkpoint_discriminator_gradients,
        precision=settings.numeric_precision.value,
        slmadv_min_len=settings.slmadv_min_len,
        slmadv_max_len=settings.slmadv_max_len,
        slmadv_batch_samples=settings.slmadv_batch_samples,
        slmadv_scale=settings.slm_scale,
        architecture_path=architecture_path,
        multispeaker=settings.multispeaker,
        decoder_type=settings.decoder.value,
        studio_publish=_studio_publish(scratch, base_checkpoint, pretrained_model, settings.display_name),
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
) -> dict[str, Any]:
    if scratch:
        return {
            "enabled": True,
            "parent_checkpoint_id": "",
            "parent_checkpoint_path": "",
            "base_library_root": "",
            "pretrained_relpath": "",
            "run_name": run_name,
        }
    return {
        "enabled": True,
        "parent_checkpoint_id": str(base_checkpoint.checkpoint_id),
        "parent_checkpoint_path": str(base_checkpoint.path),
        "base_library_root": str(base_checkpoint.path),
        "pretrained_relpath": _relative_weight(base_checkpoint.path, pretrained_model),
        "run_name": run_name,
    }


def _training_asset_paths(ref: AssetBundleRef | None) -> tuple[dict[str, Path | None], list[Path]]:
    paths: dict[str, Path | None] = {"asr_bundle": None, "f0_model": None, "plbert": None}
    ood_paths: list[Path] = []
    if ref is None:
        return paths, ood_paths
    for asset in ref.metadata["assets"]:
        role = str(asset["role"])
        if role == "ood_text_set":
            ood_paths.append(Path(str(asset["path"])))
        elif role in paths and paths[role] is None:
            paths[role] = Path(str(asset["path"]))
    return paths, ood_paths


def _ood_text_path(ood_paths: list[Path], output_dir: Path) -> Path:
    existing = [path for path in ood_paths if path.is_file()]
    if not existing:
        raise ValueError("StyleTTS finetune config requires an OOD text file path")
    if len(existing) == 1:
        return existing[0]
    combined = output_dir / "ood_texts.txt"
    combined.write_text("\n".join(path.read_text(encoding="utf-8").strip() for path in existing), encoding="utf-8")
    return combined


def _asr_config(symbol_count: int) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.ASR_YAML)
    config["model_params"]["n_token"] = int(symbol_count)
    return config


def _plbert_config(symbol_count: int) -> dict[str, Any]:
    config = styletts_config.load_yaml(styletts_config.PLBERT_YAML)
    config["model_params"]["vocab_size"] = int(symbol_count)
    return config


def _relative_weight(root: Path, weight: Path) -> str:
    return weight.resolve().relative_to(root.resolve()).as_posix()
