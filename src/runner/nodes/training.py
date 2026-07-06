from __future__ import annotations

from enum import Enum

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import JSON


class NumericPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DecoderBackend(str, Enum):
    HIFIGAN = "hifigan"
    ISTFTNET = "istftnet"


class AlphabetPreset(str, Enum):
    IPA = "ipa"
    ARPABET = "arpabet"
    IPA_MULTI = "ipa-multi"
    CUSTOM = "custom"


DEFAULT_ALPHABET = "a b c d e f g h i j k l m n o p q r s t u v w x y z ɑ æ ə ɛ ɪ ʊ ʌ ɔ θ ð ʃ ʒ ŋ tʃ dʒ aɪ aʊ eɪ oʊ ɔɪ ɝ ɚ ˈ ˌ ː . , ? ! ' \" ( ) -"


class SelectTrainingDatasetSettings(StrictSettings):
    dataset_id: str = Field(default="", title="Training dataset")


class SelectCheckpointSettings(StrictSettings):
    checkpoint_id: str = Field(default="", title="Checkpoint")


class SelectTrainingAssetsSettings(StrictSettings):
    f0_model: str = Field(default="", title="F0 model")
    asr_model: str = Field(default="", title="ASR model")
    plbert_model: str = Field(default="", title="PL-BERT")


class PhonemeAlphabetSettings(StrictSettings):
    preset: AlphabetPreset = Field(default=AlphabetPreset.IPA, title="Alphabet preset")
    symbols: str = Field(default=DEFAULT_ALPHABET, title="Symbols")


class OodTextSet(StrictSettings):
    id: str
    name: str
    line_count: int


class SelectOodTextSetsSettings(StrictSettings):
    sets: list[OodTextSet] = Field(default_factory=list, title="Reference text sets")


class ListDatasetAudioIdsSettings(StrictSettings):
    include_virtual: bool = False


class StyleTtsFinetuneSettings(StrictSettings):
    display_name: str = Field(default="styletts_finetune", title="Display name")
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
    decoder: DecoderBackend = Field(default=DecoderBackend.HIFIGAN, title="Decoder")
    slm_weight: float = Field(default=0.2, title="SLM weight", ge=0)
    diffusion_samples: int = Field(default=3, title="Diffusion samples", ge=1)
    slm_scale: float = Field(default=0.01, title="Scale", ge=0)
    multispeaker: bool = Field(default=True, title="Multi-speaker mode")
    checkpoint_each_stage: bool = Field(default=True, title="Checkpoint each stage")
    mixed_precision: bool = Field(default=False, title="Mixed precision")


class F0TrainingSettings(StrictSettings):
    display_name: str = Field(default="f0_v2", title="Display name")
    validation_samples: int = Field(default=32, title="Validation samples", ge=0, le=512)
    batch_size: int = Field(default=32, title="Batch size", ge=1, le=256)
    learning_rate: float = Field(default=5e-4, title="Learning rate", gt=0)
    epochs: int = Field(default=100, title="Epochs", ge=1)
    save_interval_epochs: int = Field(default=10, title="Save interval (epochs)", ge=1)


class AsrTrainingSettings(F0TrainingSettings):
    display_name: str = Field(default="asr_v2", title="Display name")
    dataloader_workers: int = Field(default=8, title="Dataloader workers", ge=0, le=64)


class TrainingRunInputNode(Node):
    NODE_TYPE = "TrainingRunInput"
    CATEGORY = "Training / Inputs"
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"run": Port("run", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"input node already emitted: {self.id}"
        self._emitted = True
        return [{"run": {"node_type": self.NODE_TYPE, "source": "workflow"}}]


class MockTrainingInputNode(Node):
    CATEGORY = "Training / Inputs"
    INPUTS = {"run": Port("run", JSON)}
    OUTPUT_FIELD = "input"
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{self.OUTPUT_FIELD: {"node_type": self.NODE_TYPE, "run": inputs["run"], "source": "workflow_settings"}} for inputs in batch]


class SelectTrainingDatasetNode(MockTrainingInputNode):
    NODE_TYPE = "SelectTrainingDataset"
    SETTINGS = SelectTrainingDatasetSettings
    OUTPUT_FIELD = "dataset_ref"
    OUTPUTS = {"dataset_ref": Port("dataset_ref", JSON)}


class SelectCheckpointNode(MockTrainingInputNode):
    NODE_TYPE = "SelectCheckpoint"
    CATEGORY = "Assets / Inputs"
    SETTINGS = SelectCheckpointSettings
    OUTPUT_FIELD = "checkpoint_ref"
    OUTPUTS = {"checkpoint_ref": Port("checkpoint_ref", JSON)}


class SelectTrainingAssetsNode(MockTrainingInputNode):
    NODE_TYPE = "SelectTrainingAssets"
    SETTINGS = SelectTrainingAssetsSettings
    OUTPUT_FIELD = "asset_refs"
    OUTPUTS = {"asset_refs": Port("asset_refs", JSON)}


class PhonemeAlphabetNode(MockTrainingInputNode):
    NODE_TYPE = "PhonemeAlphabet"
    CATEGORY = "Text / Inputs"
    SETTINGS = PhonemeAlphabetSettings
    OUTPUT_FIELD = "phoneme_alphabet"
    OUTPUTS = {"phoneme_alphabet": Port("phoneme_alphabet", JSON)}


class SelectOodTextSetsNode(MockTrainingInputNode):
    NODE_TYPE = "SelectOodTextSets"
    SETTINGS = SelectOodTextSetsSettings
    OUTPUT_FIELD = "ood_text_set_refs"
    OUTPUTS = {"ood_text_set_refs": Port("ood_text_set_refs", JSON)}


class ListDatasetAudioIdsNode(Node):
    NODE_TYPE = "ListDatasetAudioIds"
    CATEGORY = "Training / DB"
    SETTINGS = ListDatasetAudioIdsSettings
    INPUTS = {"dataset_ref": Port("dataset_ref", JSON)}
    OUTPUTS = {"audio_file_ids": Port("audio_file_ids", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"audio_file_ids": {"source": inputs["dataset_ref"], "include_virtual": self.settings.include_virtual, "ids": []}} for inputs in batch]


class PrefetchCheckpointNode(Node):
    NODE_TYPE = "PrefetchCheckpoint"
    CATEGORY = "Assets"
    INPUTS = {"checkpoint_ref": Port("checkpoint_ref", JSON)}
    OUTPUTS = {"checkpoint": Port("checkpoint", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"checkpoint": {"source": inputs["checkpoint_ref"], "cache": "asset"}} for inputs in batch]


class PrefetchTrainingAssetsNode(Node):
    NODE_TYPE = "PrefetchTrainingAssets"
    CATEGORY = "Training / Assets"
    INPUTS = {"asset_refs": Port("asset_refs", JSON)}
    OUTPUTS = {"assets": Port("assets", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"assets": {"source": inputs["asset_refs"], "cache": "asset"}} for inputs in batch]


class PrefetchOodTextSetsNode(Node):
    NODE_TYPE = "PrefetchOodTextSets"
    CATEGORY = "Training / DB"
    INPUTS = {"ood_text_set_refs": Port("ood_text_set_refs", JSON)}
    OUTPUTS = {"ood_text_sets": Port("ood_text_sets", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"ood_text_sets": {"source": inputs["ood_text_set_refs"], "cache": "asset"}} for inputs in batch]


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "base_checkpoint": Port("base_checkpoint", JSON),
        "pretrained_assets": Port("pretrained_assets", JSON, optional=True),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
        "ood_text_sets": Port("ood_text_sets", JSON),
    }
    OUTPUTS = {"training_result": Port("training_result", JSON)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"training_result": {"node_type": self.NODE_TYPE, "settings": self.params}} for _inputs in batch]


class F0ModelTrainingNode(Node):
    NODE_TYPE = "F0ModelTraining"
    CATEGORY = "Training"
    SETTINGS = F0TrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", JSON, optional=True),
    }
    OUTPUTS = {"training_result": Port("training_result", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 6}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"training_result": {"node_type": self.NODE_TYPE, "settings": self.params}} for _inputs in batch]


class AsrModelTrainingNode(Node):
    NODE_TYPE = "AsrModelTraining"
    CATEGORY = "Training"
    SETTINGS = AsrTrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", JSON, optional=True),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
    }
    OUTPUTS = {"training_result": Port("training_result", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"training_result": {"node_type": self.NODE_TYPE, "settings": self.params}} for _inputs in batch]
