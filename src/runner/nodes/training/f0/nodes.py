from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import CheckpointRefPort, JsonPort, TrainingResultPort
from runner.nodes.training.common.results import (
    checkpoint_weight,
    publish_training_result,
    database_training_output_dir,
)
from runner.nodes.training.common.mlflow_run import start_mlflow_run


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
    DESCRIPTION = "Train an F0 pitch model directly from a stored dataset without an intermediate manifest."
    CATEGORY = "Training"
    SETTINGS = F0TrainingSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "dataset_ref": JsonPort(),
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

    dataset_id = UUID(str(inputs["dataset_ref"]["dataset_id"]))
    output_dir = database_training_output_dir(
        settings.output_checkpoint_dir,
        run_id,
        "f0",
    )
    tracker = start_mlflow_run(
        experiment="f0_training",
        name=f"{settings.display_name}-{run_id}",
        config=settings.model_dump(mode="json"),
    )
    try:
        train_f0_model(
            run=tracker,
            dataset_id=dataset_id,
            validation_samples=settings.validation_samples,
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
        tracker.close()
        release_accelerator_memory()
    validate_f0_checkpoint_folder(output_dir)
    return publish_training_result(
        "F0ModelTraining",
        settings.display_name,
        "f0_model",
        str(output_dir),
        {"dataset_id": str(dataset_id)},
        run_id,
    )
