from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field
import torch

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import CheckpointRefPort, JsonPort, TrainingResultPort
from runner.nodes.models import typed_checkpoint
from runner.nodes.mos.model import load_base_mos_bundle
from runner.nodes.mos.train import train_mos_model
from runner.nodes.training.common.results import (
    publish_training_result,
    database_training_output_dir,
)


class MosModelTrainingSettings(StrictSettings):
    display_name: str = Field(default="mos_wav2vec2", title="Display name")
    output_checkpoint_dir: str = Field(default="", title="External output checkpoint folder")
    batch_size: int = Field(default=2, title="Batch size", ge=1, le=64)
    learning_rate: float = Field(default=1e-5, title="Learning rate", gt=0)
    weight_decay: float = Field(default=0.01, title="Weight decay", ge=0)
    comparison_weight: float = Field(default=1.0, title="Comparison loss weight", ge=0)
    epochs: int = Field(default=10, title="Epochs", ge=1)
    dataloader_workers: int = Field(default=0, title="Dataloader workers", ge=0, le=32)
    save_interval_epochs: int = Field(default=1, title="Save interval (epochs)", ge=1)
    validation_comparisons: int = Field(
        default=32,
        title="Validation comparisons",
        ge=1,
        le=10_000,
    )


class MosModelTrainingNode(Node):
    NODE_TYPE = "MosModelTraining"
    DESCRIPTION = "Fine-tune a MOS quality model directly from stored dataset comparisons without an intermediate manifest."
    CATEGORY = "Training"
    SETTINGS = MosModelTrainingSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "dataset_ref": JsonPort(),
    }
    OUTPUTS = {"training_result": TrainingResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=1, max_size=1)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 12},
        exclusive_group="accelerator",
    )
    QUEUE_MAX_SIZE = 1

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            outputs.append({"training_result": await self._train(inputs, context)})
        return outputs

    async def _train(self, inputs: dict[str, Any], context):
        checkpoint = typed_checkpoint(inputs["checkpoint"])
        if checkpoint.metadata["type"] != "mos_base":
            raise ValueError(f"MosModelTraining requires mos_base checkpoint: {checkpoint.checkpoint_id}")
        dataset_id = UUID(str(inputs["dataset_ref"]["dataset_id"]))
        output_dir = database_training_output_dir(
            self.settings.output_checkpoint_dir,
            str(context.run_id),
            "mos",
        )
        device = torch.device(str(context.device))
        bundle = load_base_mos_bundle(checkpoint.path, device)
        try:
            metrics = await train_mos_model(
                bundle=bundle,
                dataset_id=dataset_id,
                validation_comparisons=self.settings.validation_comparisons,
                output_dir=output_dir,
                device=device,
                epochs=self.settings.epochs,
                batch_size=self.settings.batch_size,
                learning_rate=self.settings.learning_rate,
                weight_decay=self.settings.weight_decay,
                comparison_weight=self.settings.comparison_weight,
                dataloader_workers=self.settings.dataloader_workers,
                save_interval_epochs=self.settings.save_interval_epochs,
                context=context,
                node_id=self.id,
            )
        finally:
            release_accelerator_memory()
        metadata = {
            "dataset_id": str(dataset_id),
            "base_checkpoint_id": str(checkpoint.checkpoint_id),
            "metrics": metrics.as_dict(),
        }
        return publish_training_result(
            self.NODE_TYPE,
            self.settings.display_name,
            "mos_model",
            str(output_dir),
            metadata,
            str(context.run_id),
        )
