from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AssetBundlePort, CheckpointRefPort, JsonPort
from runner.nodes.models import typed_assets, typed_checkpoint
from runner.nodes.training.common.run_directory import claim_run_dir, remove_run_dir
from runner.nodes.training.styletts.config import build_configs
from runner.nodes.training.styletts.launch import create_training, train
from traintts.stages import TrainingStageSpec, default_training_stages


class NumericPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DecoderBackend(str, Enum):
    HIFIGAN = "hifigan"
    ISTFTNET = "istftnet"


class StyleTtsFinetuneSettings(StrictSettings):
    display_name: str = Field(default="styletts_finetune", title="Display name")
    seed: int = Field(default=1, title="Random seed", ge=0)
    validation_samples: int = Field(default=32, title="Validation samples", ge=1, le=512)
    learning_rate: float = Field(default=1e-4, title="Learning rate", gt=0)
    numeric_precision: NumericPrecision = Field(default=NumericPrecision.FP32, title="Numeric precision")
    training_stages: list[TrainingStageSpec] = Field(
        default_factory=default_training_stages,
        title="Training stages",
        min_length=1,
    )
    validation_interval_steps: int = Field(default=1000, title="Validation interval", ge=1)
    checkpoint_interval_steps: int = Field(default=5000, title="Checkpoint interval", ge=1)
    log_interval_steps: int = Field(default=10, title="Log interval", ge=1)
    distributed_processes: int = Field(default=1, title="Distributed processes", ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=1, title="Gradient accumulation steps", ge=1)
    load_optimizer: bool = Field(default=False, title="Load optimizer state")
    reset_training_step: bool = Field(default=False, title="Reset training step")
    decoder: DecoderBackend = Field(default=DecoderBackend.HIFIGAN, title="Decoder")
    multispeaker: bool = Field(default=True, title="Multi-speaker mode")
    checkpoint_decoder_gradients: bool = Field(default=True, title="Checkpoint decoder gradients")
    checkpoint_discriminator_gradients: bool = Field(default=False, title="Checkpoint discriminator gradients")
    profiling_enabled: bool = Field(default=False, title="Training profiler")


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    DESCRIPTION = "Create a GiveMeData session from the frontend config and launch traintts."
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "dataset_ref": JsonPort(),
        "phoneme_alphabet": JsonPort(join_mode=JoinMode.BROADCAST),
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "assets": AssetBundlePort(join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"training": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 12},
        exclusive_group="accelerator",
    )

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            run_id = str(context.run_id)
            run_dir = Path("data/training/styletts") / run_id
            data_config, train_config = build_configs(
                dataset_id=str(inputs["dataset_ref"]["dataset_id"]),
                phoneme_alphabet=[
                    str(symbol)
                    for symbol in inputs["phoneme_alphabet"]["symbol_list"]
                ],
                base_checkpoint=typed_checkpoint(inputs["checkpoint"]),
                pretrained_assets=typed_assets(inputs["assets"]),
                settings=self.settings,
                output_dir=run_dir,
            )
            with claim_run_dir(run_dir):
                try:
                    run_dir.mkdir(parents=True)
                    training_id = create_training(data_config, train_config)
                    train(
                        training_id,
                        self.settings.distributed_processes,
                        self.settings.numeric_precision.value,
                        run_dir,
                    )
                finally:
                    remove_run_dir(run_dir)
            outputs.append({
                "training": {
                    "training_id": str(training_id),
                    "dataset_id": data_config["dataset_id"],
                }
            })
        return outputs
