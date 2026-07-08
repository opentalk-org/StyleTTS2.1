from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from shared.logging_setup import get_logger

logger = get_logger(__name__)

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import ASSET_BUNDLE, CHECKPOINT_REF, TRAINING_MANIFEST, TRAINING_RESULT
from runner.nodes.models import AssetBundleRef, CheckpointRef, TrainingManifest, TrainingResult, stable_id, typed_assets, typed_checkpoint
from runner.nodes.training.common.results import NoopAimRun
from runner.nodes.training.styletts.finetune.node_config import build_node_config
from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session


class NumericPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DecoderBackend(str, Enum):
    HIFIGAN = "hifigan"
    ISTFTNET = "istftnet"


class StyleTtsFinetuneSettings(StrictSettings):
    display_name: str = Field(default="styletts_finetune", title="Display name")
    output_checkpoint_dir: str = Field(default="", title="External output checkpoint folder")
    validation_samples: int = Field(default=32, title="Validation samples", ge=0, le=512)
    batch_size: int = Field(default=16, title="Batch size", ge=1, le=128)
    learning_rate: float = Field(default=1e-4, title="Learning rate", gt=0)
    numeric_precision: NumericPrecision = Field(default=NumericPrecision.BF16, title="Numeric precision")
    epochs_base: int = Field(default=30, title="Epochs · base", ge=0)
    epochs_diffusion: int = Field(default=15, title="Epochs · diffusion", ge=0)
    epochs_joint: int = Field(default=5, title="Epochs · joint", ge=0)
    max_sequence_seconds: float = Field(default=8.0, title="Max sequence (sec)", ge=1, le=30)
    save_interval_epochs: int = Field(default=5, title="Save interval (epochs)", ge=1)
    load_optimizer: bool = Field(default=False, title="Load optimizer state")
    slmadv_min_len: int = Field(default=180, title="SLM min length", ge=1)
    slmadv_max_len: int = Field(default=200, title="SLM max length", ge=1)
    slmadv_batch_samples: int = Field(default=0, title="SLM batch samples", ge=0)
    decoder: DecoderBackend = Field(default=DecoderBackend.HIFIGAN, title="Decoder")
    slm_scale: float = Field(default=0.01, title="Scale", ge=0)
    multispeaker: bool = Field(default=True, title="Multi-speaker mode")
    checkpoint_decoder_gradients: bool = Field(default=True, title="Checkpoint decoder gradients")
    checkpoint_discriminator_gradients: bool = Field(default=True, title="Checkpoint discriminator gradients")
    config_output_dir: str = Field(default="", title="Config output folder")


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST),
        "checkpoint": Port("checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST),
        "assets": Port("assets", ASSET_BUNDLE, join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            training_config = build_styletts_finetune_config(
                manifest=inputs["training_manifest"],
                base_checkpoint=typed_checkpoint(inputs["checkpoint"]),
                pretrained_assets=typed_assets(inputs["assets"]),
                settings=self.settings,
            )
            config_path = _prepare_styletts_config(training_config, str(context.run_id))
            _run_styletts_train(config_path)
            outputs.append({"training_result": _latest_epoch_result(str(context.run_id))})
        return outputs


def build_styletts_finetune_config(
    manifest: TrainingManifest,
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    settings: StyleTtsFinetuneSettings,
) -> dict[str, Any]:
    output_dir = _config_output_dir(manifest, settings.config_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, styletts_yaml = build_node_config(
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        pretrained_assets=pretrained_assets,
        settings=settings,
        output_dir=output_dir,
    )
    return {
        "version": 2,
        "node_type": StyleTtsFinetuneNode.NODE_TYPE,
        "config_path": str(config_path),
        "manifest": _manifest_payload(manifest),
        "base_checkpoint": _checkpoint_payload(base_checkpoint),
        "pretrained_assets": _assets_payload(pretrained_assets),
        "training": settings.model_dump(mode="json"),
        "styletts_yaml": styletts_yaml,
    }


def _prepare_styletts_config(training_config: dict[str, Any], run_id: str) -> Path:
    config_path = Path(str(training_config["config_path"]))
    styletts_yaml = dict(training_config["styletts_yaml"])
    Path(str(styletts_yaml["log_dir"])).mkdir(parents=True, exist_ok=True)
    publish = dict(styletts_yaml["studio_publish"])
    publish["run_id"] = run_id
    publish["finetune_job_id"] = run_id
    styletts_yaml["studio_publish"] = publish
    config_path.write_text(yaml.safe_dump(styletts_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def _run_styletts_train(config_path: Path) -> None:
    from runner.nodes.training.styletts.finetune.training.train_finetune import train

    aim_run = _make_aim_run(config_path)
    try:
        train(str(config_path), aim_run=aim_run)
    finally:
        close = getattr(aim_run, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("failed to close aim run", exc_info=True)
        release_accelerator_memory()


def _make_aim_run(config_path: Path) -> Any:
    """Create a real Aim run so the finetune metrics/samples reach the Aim UI.

    Falls back to a no-op run if Aim is unavailable so training never fails just
    because logging could not be initialized. The repo is taken from ``AIM_REPO``
    (set by the dev stack) so the run lands in the same repo the UI reads."""
    try:
        from aim import Run
    except ImportError:
        logger.warning("aim not installed; training metrics will not be logged")
        return NoopAimRun()
    try:
        styletts_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        styletts_yaml = {}
    publish = styletts_yaml.get("studio_publish", {}) if isinstance(styletts_yaml, dict) else {}
    run_name = str(publish.get("run_name") or publish.get("run_id") or "styletts_finetune")
    repo = os.environ.get("AIM_REPO") or None
    try:
        run = Run(repo=repo, experiment="styletts2_finetune")
        run.name = run_name
        run["hparams"] = {
            "run_id": publish.get("run_id"),
            "finetune_job_id": publish.get("finetune_job_id"),
            "epochs": styletts_yaml.get("epochs"),
            "batch_size": styletts_yaml.get("batch_size"),
            "max_len": styletts_yaml.get("max_len"),
            "precision": styletts_yaml.get("precision"),
            "n_token": (styletts_yaml.get("model_params") or {}).get("n_token"),
            "decoder": ((styletts_yaml.get("model_params") or {}).get("decoder") or {}).get("type"),
        }
        logger.info("aim run started name=%s repo=%s", run_name, repo or "<default>")
        return run
    except Exception:
        logger.warning("failed to start aim run; training metrics will not be logged", exc_info=True)
        return NoopAimRun()


def _latest_epoch_result(run_id: str) -> TrainingResult:
    with database_session() as session:
        checkpoints = [
            item
            for item in asset_crud.list_checkpoints(session)
            if item.metadata_.get("source") == "finetune_epoch" and item.metadata_.get("finetune_job_id") == run_id
        ]
        if not checkpoints:
            raise RuntimeError("StyleTTS finetune did not publish any epoch checkpoints")
        checkpoint = max(checkpoints, key=lambda item: item.updated_at if hasattr(item, "updated_at") else item.name)
        path = asset_crud.get_checkpoint_path(session, checkpoint.id)
    checkpoint_ref_id = stable_id("checkpoint", checkpoint.id, checkpoint.content_hash)
    checkpoint_ref = CheckpointRef(
        checkpoint_id=checkpoint.id,
        name=checkpoint.name,
        path=path,
        id=checkpoint_ref_id,
        lineage_id=checkpoint_ref_id,
        metadata={"type": checkpoint.type_, "content_hash": checkpoint.content_hash, "metadata": checkpoint.metadata_},
    )
    result_id = stable_id("training_result", "StyleTtsFinetune", run_id, checkpoint.id)
    return TrainingResult(
        training_run_id=stable_id("training_run", "StyleTtsFinetune", run_id),
        checkpoint=checkpoint_ref,
        id=result_id,
        lineage_id=result_id,
        metadata={"node_type": "StyleTtsFinetune", "checkpoint_id": str(checkpoint.id)},
    )


def _config_output_dir(manifest: TrainingManifest, configured: str) -> Path:
    if configured:
        return Path(configured)
    train_path = Path(str(manifest.metadata["train_manifest_path"]))
    return train_path.parent.parent / "configs"


def _manifest_payload(manifest: TrainingManifest) -> dict[str, Any]:
    return {
        "id": manifest.id,
        "lineage_id": manifest.lineage_id,
        "dataset_id": str(manifest.dataset_id),
        "audio_file_ids": [str(audio_id) for audio_id in manifest.audio_file_ids],
        "train_manifest_path": manifest.metadata["train_manifest_path"],
        "validation_manifest_path": manifest.metadata["validation_manifest_path"],
        "train_count": manifest.metadata["train_count"],
        "validation_count": manifest.metadata["validation_count"],
        "segment_count": manifest.metadata["segment_count"],
        "root_path": manifest.metadata["root_path"],
    }


def _checkpoint_payload(ref: CheckpointRef) -> dict[str, Any]:
    return {"id": str(ref.checkpoint_id), "name": ref.name, "path": str(ref.path), "metadata": ref.metadata}


def _assets_payload(ref: AssetBundleRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "bundle_key": ref.bundle_key,
        "name": ref.name,
        "paths": [str(path) for path in ref.paths],
        "extra_file_ids": [str(file_id) for file_id in ref.extra_file_ids],
        "metadata": ref.metadata,
    }
