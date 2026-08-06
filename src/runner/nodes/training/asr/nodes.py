from __future__ import annotations

from typing import Any
from uuid import UUID

import yaml
from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import CheckpointRefPort, JsonPort, TrainingResultPort
from runner.nodes.training.common.results import (
    checkpoint_weight,
    publish_training_result,
    database_training_output_dir,
)
from runner.nodes.training.common.mlflow_run import start_mlflow_run
from runner.nodes.training.f0.nodes import F0TrainingSettings


class AsrTrainingSettings(F0TrainingSettings):
    display_name: str = Field(default="asr_v2", title="Display name")
    dataloader_workers: int = Field(default=8, title="Dataloader workers", ge=0, le=64)


class AsrModelTrainingNode(Node):
    NODE_TYPE = "AsrModelTraining"
    DESCRIPTION = "Train an ASR text-aligner model directly from a stored dataset without an intermediate manifest."
    CATEGORY = "Training"
    SETTINGS = AsrTrainingSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "phoneme_alphabet": JsonPort(join_mode=JoinMode.BROADCAST),
        "dataset_ref": JsonPort(),
    }
    OUTPUTS = {"training_result": TrainingResultPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        return [{"training_result": _run_asr_training(self.settings, inputs, str(context.run_id))} for inputs in batch]


def _run_asr_training(settings: AsrTrainingSettings, inputs: dict[str, Any], run_id: str):
    from runner.nodes.training.asr.impl.checkpoint_publish import write_asr_bundle_artifacts
    from runner.nodes.training.asr.impl.config_load import build_effective_training_config, load_asr_reference_yaml
    from runner.nodes.training.asr.impl.train import train_asr_model

    dataset_id = UUID(str(inputs["dataset_ref"]["dataset_id"]))
    output_dir = database_training_output_dir(
        settings.output_checkpoint_dir,
        run_id,
        "asr",
    )
    symbols = _symbols(inputs["phoneme_alphabet"])
    effective = build_effective_training_config(load_asr_reference_yaml(), symbols=symbols)
    config_path = output_dir / "effective_asr_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tracker = start_mlflow_run(
        experiment="asr_training",
        name=f"{settings.display_name}-{run_id}",
        config=settings.model_dump(mode="json"),
    )
    try:
        train_asr_model(
            run=tracker,
            dataset_id=dataset_id,
            validation_samples=settings.validation_samples,
            run_dir=output_dir / "run",
            weights_dir=output_dir,
            effective_config=effective,
            effective_config_path=config_path,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            checkpoint_save_interval_epochs=settings.save_interval_epochs,
            pretrained_weights_path=checkpoint_weight(inputs["checkpoint"]),
            num_workers=settings.dataloader_workers,
        )
    finally:
        tracker.close()
        release_accelerator_memory()
    write_asr_bundle_artifacts(bundle_dir=output_dir, effective_config=effective)
    return publish_training_result(
        "AsrModelTraining",
        settings.display_name,
        "asr_bundle",
        str(output_dir),
        {"dataset_id": str(dataset_id)},
        run_id,
    )


def _symbols(value: dict[str, Any]) -> list[str]:
    symbol_list = value.get("symbol_list") if isinstance(value, dict) else None
    if isinstance(symbol_list, list):
        return [str(part) for part in symbol_list]
    symbols = value["symbols"] if "symbols" in value else value
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme_alphabet must provide symbols")
