from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.assets.checkpoints import SCRATCH_CHECKPOINT_ID, resolve_checkpoint_ref, scratch_checkpoint_ref
from runner.nodes.assets.training_assets import resolve_training_asset_bundle
from runner.nodes.datatypes import AssetBundlePort, CheckpointRefPort, JsonPort
from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS
from shared.db import database_session
from shared.db.common import one
from shared.db.datasets.models import Dataset


class AlphabetPreset(str, Enum):
    IPA = "ipa"
    ARPABET = "arpabet"
    IPA_MULTI = "ipa-multi"
    CUSTOM = "custom"


# Canonical StyleTTS2 symbol table: pad + punctuation + latin letters + IPA, all
# single characters. This matches the pretrained LJSpeech / LibriTTS / Vokan text
# embeddings (n_token = 178) and the espeak/phonemizer output. The legacy
# multi-character token alphabet is intentionally gone; a space-separated string
# cannot represent the literal space symbol, so `symbol_list` is authoritative.
DEFAULT_STYLETTS_ALPHABET = [str(symbol) for symbol in DEFAULT_STYLETTS_SYMBOLS]
DEFAULT_ALPHABET = "".join(DEFAULT_STYLETTS_ALPHABET)


def _alphabet_symbol_list(preset: "AlphabetPreset", symbols: str) -> list[str]:
    if preset in (AlphabetPreset.IPA, AlphabetPreset.IPA_MULTI):
        return list(DEFAULT_STYLETTS_ALPHABET)
    parsed = list(symbols)
    return parsed or list(DEFAULT_STYLETTS_ALPHABET)


class SelectTrainingDatasetSettings(StrictSettings):
    dataset_id: str = Field(default="", title="Training dataset")


class SelectCheckpointSettings(StrictSettings):
    checkpoint_id: str = Field(default="", title="Checkpoint")


class SelectTrainingAssetsSettings(StrictSettings):
    f0_model: str = Field(default="", title="F0 model")
    asr_model: str = Field(default="", title="ASR model")
    plbert_model: str = Field(default="", title="PL-BERT")
    ood_text_set_file_ids: list[str] = Field(default_factory=list, title="OOD text sets")


class PhonemeAlphabetSettings(StrictSettings):
    preset: AlphabetPreset = Field(default=AlphabetPreset.IPA, title="Alphabet preset")
    symbols: str = Field(default=DEFAULT_ALPHABET, title="Symbols")


class ListDatasetAudioIdsSettings(StrictSettings):
    include_virtual: bool = False


class TrainingRunInputNode(Node):
    NODE_TYPE = "TrainingRunInput"
    DESCRIPTION = "Start a training workflow by emitting a single run token that downstream training nodes consume. Takes no inputs and outputs one run object that seeds the rest of the graph. Place this as the entry point of any training workflow."
    CATEGORY = "Training"
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"run": JsonPort()}
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


class SelectTrainingDatasetNode(Node):
    NODE_TYPE = "SelectTrainingDataset"
    DESCRIPTION = "Pick which stored dataset to train on. Consumes a run token and outputs a reference to the chosen dataset for downstream nodes. Set the training dataset in this node's settings; feed its output into the node that lists dataset audio."
    CATEGORY = "Training"
    SETTINGS = SelectTrainingDatasetSettings
    INPUTS = {"run": JsonPort()}
    OUTPUTS = {"dataset_ref": JsonPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

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


class SelectCheckpointNode(Node):
    NODE_TYPE = "SelectCheckpoint"
    DESCRIPTION = "Pick a stored checkpoint to use as the starting point for training. Consumes a run token and outputs a checkpoint reference. Wire its output into training and manifest nodes that need a base checkpoint to finetune from."
    CATEGORY = "Inputs"
    SETTINGS = SelectCheckpointSettings
    INPUTS = {"run": JsonPort()}
    OUTPUTS = {"checkpoint": CheckpointRefPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        if not self.settings.checkpoint_id:
            raise ValueError("SelectCheckpoint requires checkpoint_id")
        if self.settings.checkpoint_id == SCRATCH_CHECKPOINT_ID:
            return [{"checkpoint": scratch_checkpoint_ref()} for _inputs in batch]
        return [{"checkpoint": resolve_checkpoint_ref(self.settings.checkpoint_id)} for _inputs in batch]


class SelectTrainingAssetsNode(Node):
    NODE_TYPE = "SelectTrainingAssets"
    DESCRIPTION = "Choose the auxiliary models and text sets required for StyleTTS finetuning: the ASR model, F0 model, PL-BERT model, and out-of-distribution text sets. Consumes a run token and outputs a bundle of these assets. Feed its output into StyleTTS finetuning, which needs all of these to train."
    CATEGORY = "Training"
    SETTINGS = SelectTrainingAssetsSettings
    INPUTS = {"run": JsonPort()}
    OUTPUTS = {"assets": AssetBundlePort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [
            {
                "assets": resolve_training_asset_bundle(
                    [self.settings.asr_model] if self.settings.asr_model else [],
                    self.settings.f0_model,
                    self.settings.plbert_model,
                    self.settings.ood_text_set_file_ids,
                )
            }
            for inputs in batch
        ]


class PhonemeAlphabetNode(Node):
    NODE_TYPE = "PhonemeAlphabet"
    DESCRIPTION = "Define the phoneme symbol set used for training. Consumes a run token and outputs the resolved list of symbols. Pick a preset (such as IPA or ARPABET) or provide custom symbols; feed the output into manifest building and ASR training so text is tokenized consistently."
    CATEGORY = "Inputs"
    SETTINGS = PhonemeAlphabetSettings
    INPUTS = {"run": JsonPort()}
    OUTPUTS = {"phoneme_alphabet": JsonPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        symbol_list = _alphabet_symbol_list(self.settings.preset, self.settings.symbols)
        return [
            {
                "phoneme_alphabet": {
                    "preset": self.settings.preset.value,
                    "symbols": self.settings.symbols,
                    "symbol_list": symbol_list,
                    "symbol_count": len(symbol_list),
                    "source": {"node_type": self.NODE_TYPE, "run": inputs["run"]},
                }
            }
            for inputs in batch
        ]


class ListDatasetAudioIdsNode(Node):
    NODE_TYPE = "ListDatasetAudioIds"
    DESCRIPTION = "Expand a selected dataset into the list of audio file ids to train on. Consumes a dataset reference and outputs the audio file ids belonging to that dataset. Optionally include virtual audio files; feed the output into manifest building. Fails if the dataset has no matching audio."
    CATEGORY = "Training"
    SETTINGS = ListDatasetAudioIdsSettings
    INPUTS = {"dataset_ref": JsonPort()}
    OUTPUTS = {"audio_file_ids": JsonPort()}
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


