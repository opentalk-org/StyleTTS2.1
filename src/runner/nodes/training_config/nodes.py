from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import ASSET_BUNDLE, CHECKPOINT_REF, JSON, TRAINING_MANIFEST
from runner.nodes.models import AssetBundleRef, CheckpointRef, TrainingManifest, TrainingResult, stable_id
from runner.nodes.styletts_finetune.node_config import build_node_config
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointCreate
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
    clip_total: float = Field(default=5.0, title="Total", gt=0)
    clip_diffusion: float = Field(default=1.0, title="Diffusion", gt=0)
    clip_slm: float = Field(default=0.5, title="SLM", gt=0)
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
    slm_weight: float = Field(default=0.2, title="SLM weight", ge=0)
    diffusion_samples: int = Field(default=3, title="Diffusion samples", ge=1)
    slm_scale: float = Field(default=0.01, title="Scale", ge=0)
    multispeaker: bool = Field(default=True, title="Multi-speaker mode")
    checkpoint_each_stage: bool = Field(default=True, title="Checkpoint each stage")
    mixed_precision: bool = Field(default=False, title="Mixed precision")


class BuildStyleTtsFinetuneConfigSettings(StyleTtsFinetuneSettings):
    config_output_dir: str = Field(default="", title="Config output folder")


class F0TrainingSettings(StrictSettings):
    display_name: str = Field(default="f0_v2", title="Display name")
    output_checkpoint_dir: str = Field(default="", title="External output checkpoint folder")
    validation_samples: int = Field(default=32, title="Validation samples", ge=0, le=512)
    batch_size: int = Field(default=32, title="Batch size", ge=1, le=256)
    learning_rate: float = Field(default=5e-4, title="Learning rate", gt=0)
    epochs: int = Field(default=100, title="Epochs", ge=1)
    save_interval_epochs: int = Field(default=10, title="Save interval (epochs)", ge=1)
    lambda_f0: float = Field(default=0.1, title="F0 loss weight", gt=0)
    weight_decay: float = Field(default=5e-4, title="Weight decay", ge=0)
    pct_start: float = Field(default=0.0, title="Scheduler warmup pct", ge=0, le=1)
    num_workers: int = Field(default=2, title="Dataloader workers", ge=0, le=64)


class AsrTrainingSettings(F0TrainingSettings):
    display_name: str = Field(default="asr_v2", title="Display name")
    dataloader_workers: int = Field(default=8, title="Dataloader workers", ge=0, le=64)


class BuildStyleTtsFinetuneConfigNode(Node):
    NODE_TYPE = "BuildStyleTtsFinetuneConfig"
    CATEGORY = "Training / Preparation"
    SETTINGS = BuildStyleTtsFinetuneConfigSettings
    INPUTS = {
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF),
        "pretrained_assets": Port("pretrained_assets", ASSET_BUNDLE),
        "ood_text_sets": Port("ood_text_sets", JSON),
    }
    OUTPUTS = {"training_config": Port("training_config", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [
            {
                "training_config": build_styletts_finetune_config(
                    manifest=inputs["training_manifest"],
                    base_checkpoint=_typed_checkpoint(inputs["base_checkpoint"]),
                    pretrained_assets=_typed_assets(inputs["pretrained_assets"]),
                    ood_text_sets=inputs["ood_text_sets"],
                    settings=self.settings,
                )
            }
            for inputs in batch
        ]


def build_styletts_finetune_config(
    manifest: TrainingManifest,
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    ood_text_sets: dict[str, Any],
    settings: BuildStyleTtsFinetuneConfigSettings,
) -> dict[str, Any]:
    output_dir = _config_output_dir(manifest, settings.config_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, styletts_yaml = build_node_config(
        manifest=manifest,
        base_checkpoint=base_checkpoint,
        pretrained_assets=pretrained_assets,
        ood_text_sets=ood_text_sets,
        settings=settings,
        output_dir=output_dir,
    )
    return {
        "version": 2,
        "node_type": BuildStyleTtsFinetuneConfigNode.NODE_TYPE,
        "config_path": str(config_path),
        "manifest": _manifest_payload(manifest),
        "base_checkpoint": _checkpoint_payload(base_checkpoint),
        "pretrained_assets": _assets_payload(pretrained_assets),
        "ood_text_sets": ood_text_sets,
        "training": settings.model_dump(mode="json"),
        "styletts_yaml": styletts_yaml,
    }


def publish_training_result(
    node_type: str,
    display_name: str,
    checkpoint_type: str,
    output_checkpoint_dir: str,
    metadata: dict[str, Any],
    run_id: str,
) -> TrainingResult:
    if not output_checkpoint_dir:
        raise RuntimeError(f"{node_type} requires training output folder")
    folder_path = Path(output_checkpoint_dir)
    if not folder_path.is_dir():
        raise RuntimeError(f"{node_type} output_checkpoint_dir must be an existing folder: {folder_path}")

    with database_session() as session:
        checkpoint = asset_crud.create_checkpoint(
            session,
            CheckpointCreate(
                name=display_name,
                folder_path=folder_path,
                type_=checkpoint_type,
                metadata=metadata,
                job_id=run_id,
            ),
        )
        canonical_path = asset_crud.get_checkpoint_path(session, checkpoint.id)

    checkpoint_ref_id = stable_id("checkpoint", checkpoint.id, checkpoint.content_hash)
    checkpoint_ref = CheckpointRef(
        checkpoint_id=checkpoint.id,
        name=checkpoint.name,
        path=canonical_path,
        id=checkpoint_ref_id,
        lineage_id=checkpoint_ref_id,
        metadata={
            "type": checkpoint.type_,
            "size": checkpoint.size,
            "content_hash": checkpoint.content_hash,
            "job_id": checkpoint.job_id,
            "metadata": checkpoint.metadata_,
        },
    )
    result_id = stable_id("training_result", node_type, run_id, checkpoint.id)
    return TrainingResult(
        training_run_id=stable_id("training_run", node_type, run_id, checkpoint.id),
        checkpoint=checkpoint_ref,
        id=result_id,
        lineage_id=result_id,
        metadata={"node_type": node_type, "checkpoint_type": checkpoint_type, "checkpoint_id": str(checkpoint.id)},
    )


def training_config_output_dir(training_config: dict[str, Any]) -> str:
    training = training_config["training"]
    return str(training["output_checkpoint_dir"])


def _typed_checkpoint(value: CheckpointRef | dict[str, Any]) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("BuildStyleTtsFinetuneConfig requires a resolved CheckpointRef for base_checkpoint")


def _typed_assets(value: AssetBundleRef | dict[str, Any] | None) -> AssetBundleRef | None:
    if value is None or isinstance(value, AssetBundleRef):
        return value
    raise TypeError("BuildStyleTtsFinetuneConfig requires resolved AssetBundleRef values for pretrained_assets")


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
