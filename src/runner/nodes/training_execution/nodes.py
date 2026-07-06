from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import CHECKPOINT_REF, JSON, TRAINING_MANIFEST, TRAINING_RESULT
from runner.nodes.models import CheckpointRef, TrainingManifest, stable_id
from runner.nodes.training_config import (
    AsrTrainingSettings,
    F0TrainingSettings,
    StyleTtsFinetuneSettings,
    publish_training_result,
)

from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session


class NoopAimRun:
    def track(self, *args, **kwargs) -> None:
        return None


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF),
        "pretrained_assets": Port("pretrained_assets", JSON),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
        "ood_text_sets": Port("ood_text_sets", JSON),
        "training_config": Port("training_config", JSON),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            config_path = _prepare_styletts_config(inputs["training_config"], str(context.run_id))
            _run_styletts_train(config_path)
            result = _latest_epoch_result(str(context.run_id))
            outputs.append({"training_result": result})
        return outputs


class F0ModelTrainingNode(Node):
    NODE_TYPE = "F0ModelTraining"
    CATEGORY = "Training"
    SETTINGS = F0TrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", CHECKPOINT_REF),
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 6}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"training_result": _run_f0_training(self.settings, inputs, str(context.run_id))} for inputs in batch]


class AsrModelTrainingNode(Node):
    NODE_TYPE = "AsrModelTraining"
    CATEGORY = "Training"
    SETTINGS = AsrTrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", CHECKPOINT_REF),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"training_result": _run_asr_training(self.settings, inputs, str(context.run_id))} for inputs in batch]


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
    from runner.nodes.styletts_finetune.training.train_finetune import train

    train(str(config_path), aim_run=NoopAimRun())


def _run_f0_training(settings: F0TrainingSettings, inputs: dict[str, Any], run_id: str):
    from runner.nodes.f0_model_training.checkpoint_publish import validate_f0_checkpoint_folder
    from runner.nodes.f0_model_training.train import train_f0_model

    manifest: TrainingManifest = inputs["training_manifest"]
    output_dir = _training_output_dir(settings.output_checkpoint_dir, manifest, "f0")
    train_f0_model(
        aim_run=NoopAimRun(),
        train_list_path=str(manifest.metadata["train_manifest_path"]),
        val_list_path=str(manifest.metadata["validation_manifest_path"]),
        run_dir=output_dir / "run",
        weights_dir=output_dir,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        lambda_f0=settings.lambda_f0,
        checkpoint_save_interval_epochs=settings.save_interval_epochs,
        pretrained_pth=_checkpoint_weight(inputs["pretrained_checkpoint"]),
        weight_decay=settings.weight_decay,
        pct_start=settings.pct_start,
        num_workers=settings.num_workers,
    )
    validate_f0_checkpoint_folder(output_dir)
    return publish_training_result("F0ModelTraining", settings.display_name, "f0_model", str(output_dir), _metadata(inputs), run_id)


def _run_asr_training(settings: AsrTrainingSettings, inputs: dict[str, Any], run_id: str):
    from runner.nodes.asr_model_training.checkpoint_publish import write_asr_bundle_artifacts
    from runner.nodes.asr_model_training.config_load import build_effective_training_config, load_asr_reference_yaml
    from runner.nodes.asr_model_training.train import train_asr_model

    manifest: TrainingManifest = inputs["training_manifest"]
    output_dir = _training_output_dir(settings.output_checkpoint_dir, manifest, "asr")
    symbols = _symbols(inputs["phoneme_alphabet"])
    effective = build_effective_training_config(load_asr_reference_yaml(), symbols=symbols)
    config_path = output_dir / "effective_asr_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True), encoding="utf-8")
    train_asr_model(
        aim_run=NoopAimRun(),
        train_list_path=str(manifest.metadata["train_manifest_path"]),
        val_list_path=str(manifest.metadata["validation_manifest_path"]),
        run_dir=output_dir / "run",
        weights_dir=output_dir,
        effective_config=effective,
        effective_config_path=config_path,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        checkpoint_save_interval_epochs=settings.save_interval_epochs,
        pretrained_weights_path=_checkpoint_weight(inputs["pretrained_checkpoint"]),
        num_workers=settings.dataloader_workers,
    )
    write_asr_bundle_artifacts(bundle_dir=output_dir, effective_config=effective)
    return publish_training_result("AsrModelTraining", settings.display_name, "asr_bundle", str(output_dir), _metadata(inputs), run_id)


def _training_output_dir(configured: str, manifest: TrainingManifest, name: str) -> Path:
    if configured:
        output_dir = Path(configured)
    else:
        output_dir = Path(str(manifest.metadata["train_manifest_path"])).parent.parent / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _checkpoint_weight(ref: CheckpointRef) -> str | None:
    weights = sorted(ref.path.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not weights:
        return None
    return str(weights[0])


def _symbols(value: dict[str, Any]) -> list[str]:
    symbols = value["symbols"] if "symbols" in value else value
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme_alphabet must provide symbols")


def _latest_epoch_result(run_id: str):
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
    from runner.nodes.models import TrainingResult

    result_id = stable_id("training_result", "StyleTtsFinetune", run_id, checkpoint.id)
    return TrainingResult(
        training_run_id=stable_id("training_run", "StyleTtsFinetune", run_id),
        checkpoint=checkpoint_ref,
        id=result_id,
        lineage_id=result_id,
        metadata={"node_type": "StyleTtsFinetune", "checkpoint_id": str(checkpoint.id)},
    )


def _metadata(inputs: dict[str, Any]) -> dict[str, Any]:
    manifest: TrainingManifest = inputs["training_manifest"]
    return {"training_manifest": {"id": manifest.id, "dataset_id": str(manifest.dataset_id), "metadata": manifest.metadata}}
