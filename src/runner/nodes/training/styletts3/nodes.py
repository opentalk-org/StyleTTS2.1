from __future__ import annotations

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AssetBundlePort, CheckpointRefPort, TrainingManifestPort, TrainingResultPort


class StyleTts3TrainingSettings(StrictSettings):
    display_name: str = Field(default="styletts3", title="Display name")
    output_checkpoint_dir: str = Field(default="", title="External output checkpoint folder")
    batch_size: int = Field(default=16, title="Batch size", ge=1, le=128)
    learning_rate: float = Field(default=1e-4, title="Learning rate", gt=0)
    epochs: int = Field(default=30, title="Epochs", ge=1)
    save_interval_epochs: int = Field(default=5, title="Save interval (epochs)", ge=1)


class StyleTts3TrainingNode(Node):
    NODE_TYPE = "StyleTts3Training"
    DESCRIPTION = "Train a StyleTTS3 voice model on your dataset. Consumes a training manifest, a base checkpoint to start from, and an asset bundle, and produces a training result pointing at the published checkpoint. Scaffold: the training loop is not implemented yet."
    CATEGORY = "Training"
    SETTINGS = StyleTts3TrainingSettings
    INPUTS = {
        "training_manifest": TrainingManifestPort(),
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "assets": AssetBundlePort(join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"training_result": TrainingResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        raise NotImplementedError("StyleTts3Training is a scaffold; the training loop is not implemented yet")
