from __future__ import annotations

from typing import Any

import yaml
from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import CHECKPOINT_REF, JSON, TRAINING_MANIFEST, TRAINING_RESULT
from runner.nodes.models import TrainingManifest
from runner.nodes.training.common.results import (
    NoopAimRun,
    checkpoint_weight,
    publish_training_result,
    training_manifest_metadata,
    training_output_dir,
)
from runner.nodes.training.f0.nodes import F0TrainingSettings


class AsrTrainingSettings(F0TrainingSettings):
    display_name: str = Field(default="asr_v2", title="Display name")
    dataloader_workers: int = Field(default=8, title="Dataloader workers", ge=0, le=64)


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


def _run_asr_training(settings: AsrTrainingSettings, inputs: dict[str, Any], run_id: str):
    from runner.nodes.training.asr.impl.checkpoint_publish import write_asr_bundle_artifacts
    from runner.nodes.training.asr.impl.config_load import build_effective_training_config, load_asr_reference_yaml
    from runner.nodes.training.asr.impl.train import train_asr_model

    manifest: TrainingManifest = inputs["training_manifest"]
    output_dir = training_output_dir(settings.output_checkpoint_dir, manifest, "asr")
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
        pretrained_weights_path=checkpoint_weight(inputs["pretrained_checkpoint"]),
        num_workers=settings.dataloader_workers,
    )
    write_asr_bundle_artifacts(bundle_dir=output_dir, effective_config=effective)
    return publish_training_result(
        "AsrModelTraining",
        settings.display_name,
        "asr_bundle",
        str(output_dir),
        training_manifest_metadata(inputs),
        run_id,
    )


def _symbols(value: dict[str, Any]) -> list[str]:
    symbols = value["symbols"] if "symbols" in value else value
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme_alphabet must provide symbols")
