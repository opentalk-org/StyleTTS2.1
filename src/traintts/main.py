"""Standalone StyleTTS finetuning entry point.

Replicates what the training workflow's node graph used to assemble, without
the runner/backend/DB machinery. Data comes from the givemedata service
(GIVEMEDATA_ADDR, default localhost:8181); named assets are downloaded through
the same service; checkpoints land under <output_dir>/run/published_checkpoints;
metrics and artifacts go to <output_dir>/run/metrics.jsonl and artifacts/
(the MLflow integration is commented out for now).

The run spec (RunSpec yaml) is not a local file anymore: it is fetched from the
givemedata service, which passes its --train-config file through verbatim.

Usage:
    python -m traintts.main [--dry-run]

Multi-GPU:
    accelerate launch -m traintts.distributed <output_dir>/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from givemedata_client.client import GiveMeDataClient
from pydantic import BaseModel, ConfigDict, Field

from .build_config import ASR_YAML, PLBERT_YAML, build_config, load_yaml, write_config
from .config import TrainingConfig
from .default_stages import build_default_training_stages
from .layout import architecture_yaml, latest_weight
from .stages import TrainingStageSpec
from .symbols import DEFAULT_STYLETTS_SYMBOLS

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    output_dir: str
    run_name: str = "finetune"

    # local folder with config.yml + *.pth; null trains from scratch
    base_checkpoint: str | None = None
    # auxiliary models; all three are required for from-scratch training
    asr_model: str | None = None
    f0_model: str | None = None
    plbert: str | None = None
    # only used for a .safetensors plbert (was the asset's db metadata)
    plbert_symbols: list[str] | None = None
    plbert_languages: list[str] = Field(default_factory=list)

    # mlflow_run_id: str = ""  # mlflow disabled for now
    seed: int = 1
    validation_samples: int = 32
    learning_rate: float = 1e-4
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    device: str = "cuda"
    validation_every_steps: int = 500
    checkpoint_every_steps: int = 2000
    log_every_steps: int = 10
    profiling_enabled: bool = False
    distributed_processes: int = 1
    load_optimizer: bool = False
    reset_training_step: bool = False
    decoder: Literal["hifigan", "istftnet"] = "hifigan"
    multispeaker: bool = True
    checkpoint_decoder_gradients: bool = True
    checkpoint_discriminator_gradients: bool = False
    symbols: list[str] | None = None
    training_stages: list[TrainingStageSpec] | None = None


def build_run_config(spec: RunSpec) -> dict[str, Any]:
    symbols = spec.symbols or list(DEFAULT_STYLETTS_SYMBOLS)
    stages = spec.training_stages or build_default_training_stages()
    output_dir = Path(spec.output_dir)

    base = Path(spec.base_checkpoint) if spec.base_checkpoint else None
    if base is not None:
        architecture_path = architecture_yaml(base)
        pretrained_model = latest_weight(base)
    else:
        architecture_path = None
        pretrained_model = None
        missing = [
            name
            for name, value in (
                ("asr_model", spec.asr_model),
                ("f0_model", spec.f0_model),
                ("plbert", spec.plbert),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"from-scratch training requires pretrained assets: {', '.join(missing)}"
            )

    asr_path = Path(spec.asr_model) if spec.asr_model else None
    f0_path = Path(spec.f0_model) if spec.f0_model else None
    plbert_path = Path(spec.plbert) if spec.plbert else None

    config = build_config(
        log_dir=output_dir / "run",
        dataset_id=spec.dataset_id,
        validation_samples=spec.validation_samples,
        pretrained_model=pretrained_model,
        asr_config=_asr_config(len(symbols)),
        asr_path=asr_path,
        f0_path=f0_path,
        plbert_config=_plbert_config(symbols, plbert_path, spec),
        plbert_path=plbert_path,
        total_steps=sum(stage.steps for stage in stages),
        seed=spec.seed,
        learning_rate=spec.learning_rate,
        training_stages=stages,
        validation_every_steps=spec.validation_every_steps,
        checkpoint_every_steps=spec.checkpoint_every_steps,
        log_every_steps=spec.log_every_steps,
        profiling_enabled=spec.profiling_enabled,
        distributed_processes=spec.distributed_processes,
        load_optimizer=spec.load_optimizer,
        reset_training_step=spec.reset_training_step,
        generator_checkpointing=spec.checkpoint_decoder_gradients,
        discriminators_checkpointing=spec.checkpoint_discriminator_gradients,
        precision=spec.precision,
        architecture_path=architecture_path,
        multispeaker=spec.multispeaker,
        decoder_type=spec.decoder,
        studio_publish={
            "enabled": False,
            "parent_checkpoint_id": "",
            "parent_checkpoint_path": str(base) if base else "",
            "base_library_root": str(base) if base else "",
            "pretrained_relpath": "",
            "run_id": spec.run_name,
            "finetune_job_id": spec.run_name,
            "run_name": spec.run_name,
            "mlflow_run_id": "",  # mlflow disabled for now
        },
        symbols=symbols,
        symbol_count=len(symbols),
    )
    config["device"] = spec.device
    return config


def _asr_config(symbol_count: int) -> dict[str, Any]:
    config = load_yaml(ASR_YAML)
    config["model_params"]["n_token"] = int(symbol_count)
    return config


def _plbert_config(
    symbols: list[str],
    path: Path | None,
    spec: RunSpec,
) -> dict[str, Any]:
    config = load_yaml(PLBERT_YAML)
    config["model_params"]["vocab_size"] = len(symbols)
    if path is not None and path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            positions = checkpoint.get_slice(
                "encoder._orig_mod.embeddings.position_embeddings.weight"
            ).get_shape()[0]
        config["model_params"]["max_position_embeddings"] = int(positions)
        config["input_symbols"] = symbols
        config["artifact_symbols"] = spec.plbert_symbols or symbols
        config["languages"] = spec.plbert_languages
        config["modality_id"] = 0
    return config


def _resolve_assets(spec: RunSpec, client: GiveMeDataClient) -> None:
    """Turn asset names from the train config into local paths, downloading
    through the givemedata Asset RPC (skipped when already cached on disk)."""
    assets_dir = Path(os.environ.get("TRAINTTS_ASSETS_DIR", ".cache/traintts/assets"))
    for field in ("asr_model", "f0_model", "plbert"):
        name = getattr(spec, field)
        if name:
            path = client.download_asset(name, assets_dir)
            logger.info("asset %s (%s) -> %s", field, name, path)
            setattr(spec, field, str(path))
    if spec.base_checkpoint:
        path = client.download_asset(spec.base_checkpoint, assets_dir)
        logger.info("asset base_checkpoint (%s) -> %s", spec.base_checkpoint, path)
        spec.base_checkpoint = str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="traintts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the full training config, print it, and exit",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    data_client = GiveMeDataClient()
    logger.info("fetched train config from givemedata session=%s", data_client.session_id)
    spec = RunSpec.model_validate(yaml.safe_load(data_client.train_config))
    if not arguments.dry_run:
        # dry-run keeps the asset names as-is; nothing is downloaded
        _resolve_assets(spec, data_client)
    config = build_run_config(spec)
    training_config = TrainingConfig.model_validate(config)

    if arguments.dry_run:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        data_client.close()
        return

    output_dir = Path(spec.output_dir)
    config_path = output_dir / "config.yaml"
    write_config(config_path, config)
    Path(training_config.log_dir).mkdir(parents=True, exist_ok=True)
    logger.info("run %r starting, resolved config written to %s", spec.run_name, config_path)

    # deferred: these pull in torch and the whole model stack
    from .mlflow_logging import start_run
    from .train import train

    run = start_run(training_config)
    try:
        train(str(config_path), run=run, data_client=data_client)
    finally:
        run.close()


if __name__ == "__main__":
    main()
