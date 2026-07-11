from __future__ import annotations

from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import CheckpointRefPort, TrainingManifestPort, TrainingResultPort
from runner.nodes.models import TrainingManifest
from runner.nodes.training.common.results import (
    NoopAimRun,
    checkpoint_weight,
    publish_training_result,
    training_manifest_metadata,
    training_output_dir,
)


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


class F0ModelTrainingNode(Node):
    NODE_TYPE = "F0ModelTraining"
    DESCRIPTION = "Train an F0 (pitch) prediction model used by StyleTTS from a training manifest, starting from a pretrained checkpoint. Consumes a checkpoint and a training manifest, and produces a training result pointing at the trained F0 model. Use it to produce the pitch model that StyleTTS finetuning depends on for a given dataset."
    CATEGORY = "Training"
    SETTINGS = F0TrainingSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "training_manifest": TrainingManifestPort(),
    }
    OUTPUTS = {"training_result": TrainingResultPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 6}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        return [{"training_result": _run_f0_training(self.settings, inputs, str(context.run_id))} for inputs in batch]


def _run_f0_training(settings: F0TrainingSettings, inputs: dict[str, Any], run_id: str):
    from runner.nodes.training.f0.impl.checkpoint_publish import validate_f0_checkpoint_folder
    from runner.nodes.training.f0.impl.train import train_f0_model

    manifest: TrainingManifest = inputs["training_manifest"]
    output_dir = training_output_dir(settings.output_checkpoint_dir, manifest, "f0")
    try:
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
            pretrained_pth=checkpoint_weight(inputs["checkpoint"]),
            weight_decay=settings.weight_decay,
            pct_start=settings.pct_start,
            num_workers=settings.num_workers,
        )
    finally:
        release_accelerator_memory()
    validate_f0_checkpoint_folder(output_dir)
    return publish_training_result(
        "F0ModelTraining",
        settings.display_name,
        "f0_model",
        str(output_dir),
        training_manifest_metadata(inputs),
        run_id,
    )
