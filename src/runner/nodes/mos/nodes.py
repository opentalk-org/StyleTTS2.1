from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field
import torch

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import CheckpointRefPort, JsonPort, TrainingManifestPort, TrainingResultPort
from runner.nodes.models import TrainingManifest, typed_checkpoint
from runner.nodes.mos.manifest import build_mos_training_manifest
from runner.nodes.mos.model import load_base_mos_bundle
from runner.nodes.mos.train import train_mos_model
from runner.nodes.training.common.results import (
    publish_training_result,
    training_manifest_metadata,
    training_output_dir,
)


DEFAULT_MOS_MANIFEST_DIR = Path("data/training/mos_manifests")


class BuildMosTrainingManifestSettings(StrictSettings):
    validation_comparisons: int = Field(default=32, title="Validation comparisons", ge=1, le=10_000)
    output_dir: Path = Field(default=DEFAULT_MOS_MANIFEST_DIR, title="Output directory")


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


class BuildMosTrainingManifestNode(Node):
    NODE_TYPE = "BuildMosTrainingManifest"
    DESCRIPTION = "Assemble a training manifest for MOS-model training from a dataset reference and a checkpoint. It gathers the labelled audio for the dataset and builds train and validation splits (including a configurable number of validation comparison pairs), writing the manifest to disk. Feed its output into the MOS model training node."
    CATEGORY = "Training"
    SETTINGS = BuildMosTrainingManifestSettings
    INPUTS = {
        "dataset_ref": JsonPort(),
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"training_manifest": TrainingManifestPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 4

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            context.check_cancel()
            dataset_id = UUID(str(inputs["dataset_ref"]["dataset_id"]))
            output_dir = self.settings.output_dir
            if output_dir == DEFAULT_MOS_MANIFEST_DIR:
                output_dir = output_dir / str(context.run_id)
            manifest = build_mos_training_manifest(
                dataset_id,
                typed_checkpoint(inputs["checkpoint"]),
                self.settings.validation_comparisons,
                output_dir,
            )
            outputs.append({"training_manifest": manifest})
        return outputs


class MosModelTrainingNode(Node):
    NODE_TYPE = "MosModelTraining"
    DESCRIPTION = "Fine-tune a MOS quality-prediction model from a base MOS checkpoint and a training manifest, producing a trained MOS-model checkpoint as a training result. Configure epochs, batch size, learning rate, and the comparison-loss weight in the settings, and optionally direct the output checkpoint to an external folder. Use it to train the model consumed by the MOS scoring node."
    CATEGORY = "Training"
    SETTINGS = MosModelTrainingSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "training_manifest": TrainingManifestPort(),
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
        manifest = inputs["training_manifest"]
        if not isinstance(manifest, TrainingManifest):
            raise TypeError("MosModelTraining requires TrainingManifest input")
        output_dir = training_output_dir(
            self.settings.output_checkpoint_dir,
            manifest,
            "mos",
        )
        device = torch.device(str(context.device))
        bundle = load_base_mos_bundle(checkpoint.path, device)
        try:
            metrics = await train_mos_model(
                bundle=bundle,
                manifest=manifest,
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
            **training_manifest_metadata(inputs),
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
