from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.assets.checkpoints import checkpoint_ref_or_stub, prefetch_checkpoint_ref
from runner.nodes.assets.training_assets import prefetch_training_assets, training_assets_ref_or_stub
from runner.nodes.datatypes import JSON
from runner.nodes.training_config import ASSET_BUNDLE_OR_JSON, CHECKPOINT_REF_OR_JSON
from shared.db import database_session
from shared.db.common import one
from shared.db.datasets.models import Dataset


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

    async def execute(self, batch, context):
        if not self.settings.dataset_id:
            raise ValueError("SelectTrainingDataset requires dataset_id")
        dataset_id = UUID(self.settings.dataset_id)
        return [
            {
                "dataset_ref": {
                    "dataset_id": str(dataset_id),
                    "node_type": self.NODE_TYPE,
                    "run": inputs["run"],
                    "source": "workflow_settings",
                }
            }
            for inputs in batch
        ]


class SelectCheckpointNode(MockTrainingInputNode):
    NODE_TYPE = "SelectCheckpoint"
    CATEGORY = "Assets / Inputs"
    SETTINGS = SelectCheckpointSettings
    OUTPUT_FIELD = "checkpoint_ref"
    OUTPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF_OR_JSON)}

    async def execute(self, batch, context):
        return [
            {"checkpoint_ref": checkpoint_ref_or_stub(self.NODE_TYPE, inputs["run"], self.settings.checkpoint_id)}
            for inputs in batch
        ]


class SelectTrainingAssetsNode(MockTrainingInputNode):
    NODE_TYPE = "SelectTrainingAssets"
    SETTINGS = SelectTrainingAssetsSettings
    OUTPUT_FIELD = "asset_refs"
    OUTPUTS = {"asset_refs": Port("asset_refs", ASSET_BUNDLE_OR_JSON)}

    async def execute(self, batch, context):
        return [
            {
                "asset_refs": training_assets_ref_or_stub(
                    self.NODE_TYPE,
                    inputs["run"],
                    self.settings.f0_model,
                    self.settings.asr_model,
                    self.settings.plbert_model,
                )
            }
            for inputs in batch
        ]


class PhonemeAlphabetNode(MockTrainingInputNode):
    NODE_TYPE = "PhonemeAlphabet"
    CATEGORY = "Text / Inputs"
    SETTINGS = PhonemeAlphabetSettings
    OUTPUT_FIELD = "phoneme_alphabet"
    OUTPUTS = {"phoneme_alphabet": Port("phoneme_alphabet", JSON)}

    async def execute(self, batch, context):
        return [
            {
                "phoneme_alphabet": {
                    "preset": self.settings.preset.value,
                    "symbols": self.settings.symbols,
                    "symbol_list": [symbol for symbol in self.settings.symbols.split(" ") if symbol],
                    "source": {"node_type": self.NODE_TYPE, "run": inputs["run"]},
                }
            }
            for inputs in batch
        ]


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
        outputs = []
        with database_session() as session:
            for inputs in batch:
                dataset_ref = inputs["dataset_ref"]
                dataset_id = UUID(str(dataset_ref["dataset_id"]))
                dataset = one(session, Dataset, dataset_id)
                ids = [str(item.id) for item in dataset.audio_files if self.settings.include_virtual or not item.virtual]
                if not ids:
                    raise ValueError(f"training dataset has no audio files: {dataset_id}")
                outputs.append({
                    "audio_file_ids": {
                        "source": dataset_ref,
                        "dataset_id": str(dataset_id),
                        "include_virtual": self.settings.include_virtual,
                        "ids": ids,
                    }
                })
        return outputs


class PrefetchCheckpointNode(Node):
    NODE_TYPE = "PrefetchCheckpoint"
    CATEGORY = "Assets"
    INPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF_OR_JSON)}
    OUTPUTS = {"checkpoint": Port("checkpoint", CHECKPOINT_REF_OR_JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"checkpoint": prefetch_checkpoint_ref(inputs["checkpoint_ref"])} for inputs in batch]


class PrefetchTrainingAssetsNode(Node):
    NODE_TYPE = "PrefetchTrainingAssets"
    CATEGORY = "Training / Assets"
    INPUTS = {"asset_refs": Port("asset_refs", ASSET_BUNDLE_OR_JSON)}
    OUTPUTS = {"assets": Port("assets", ASSET_BUNDLE_OR_JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"assets": prefetch_training_assets(inputs["asset_refs"])} for inputs in batch]


class PrefetchOodTextSetsNode(Node):
    NODE_TYPE = "PrefetchOodTextSets"
    CATEGORY = "Training / DB"
    INPUTS = {"ood_text_set_refs": Port("ood_text_set_refs", JSON)}
    OUTPUTS = {"ood_text_sets": Port("ood_text_sets", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"ood_text_sets": {"source": inputs["ood_text_set_refs"], "cache": "asset"}} for inputs in batch]
