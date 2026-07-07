from __future__ import annotations

from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.assets.checkpoints import prefetch_checkpoint_ref, resolve_checkpoint_ref
from runner.nodes.assets.training_assets import prefetch_training_assets, resolve_training_asset_bundle
from runner.nodes.datatypes import ASSET_BUNDLE, CHECKPOINT_REF, JSON
from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS
from shared.db import database_session
from shared.db.assets import crud as asset_crud
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
    CATEGORY = "Training"
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


class SelectTrainingDatasetNode(Node):
    NODE_TYPE = "SelectTrainingDataset"
    CATEGORY = "Training"
    SETTINGS = SelectTrainingDatasetSettings
    INPUTS = {"run": Port("run", JSON)}
    OUTPUTS = {"dataset_ref": Port("dataset_ref", JSON)}
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
    CATEGORY = "Inputs"
    SETTINGS = SelectCheckpointSettings
    INPUTS = {"run": Port("run", JSON)}
    OUTPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        if not self.settings.checkpoint_id:
            raise ValueError("SelectCheckpoint requires checkpoint_id")
        return [{"checkpoint_ref": resolve_checkpoint_ref(self.settings.checkpoint_id)} for _inputs in batch]


class SelectTrainingAssetsNode(Node):
    NODE_TYPE = "SelectTrainingAssets"
    CATEGORY = "Training"
    SETTINGS = SelectTrainingAssetsSettings
    INPUTS = {"run": Port("run", JSON)}
    OUTPUTS = {"asset_refs": Port("asset_refs", ASSET_BUNDLE)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [
            {
                "asset_refs": resolve_training_asset_bundle(
                    [self.settings.asr_model] if self.settings.asr_model else [],
                    self.settings.f0_model,
                    self.settings.plbert_model,
                    [],
                )
            }
            for inputs in batch
        ]


class PhonemeAlphabetNode(Node):
    NODE_TYPE = "PhonemeAlphabet"
    CATEGORY = "Inputs"
    SETTINGS = PhonemeAlphabetSettings
    INPUTS = {"run": Port("run", JSON)}
    OUTPUTS = {"phoneme_alphabet": Port("phoneme_alphabet", JSON)}
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


class SelectOodTextSetsNode(Node):
    NODE_TYPE = "SelectOodTextSets"
    CATEGORY = "Training"
    SETTINGS = SelectOodTextSetsSettings
    INPUTS = {"run": Port("run", JSON)}
    OUTPUTS = {"ood_text_set_refs": Port("ood_text_set_refs", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        selected = _resolve_ood_text_sets(self.settings.sets)
        return [{"ood_text_set_refs": {**selected, "run": inputs["run"]}} for inputs in batch]


class ListDatasetAudioIdsNode(Node):
    NODE_TYPE = "ListDatasetAudioIds"
    CATEGORY = "Training"
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
    INPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF)}
    OUTPUTS = {"checkpoint": Port("checkpoint", CHECKPOINT_REF)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"checkpoint": prefetch_checkpoint_ref(inputs["checkpoint_ref"])} for inputs in batch]


class PrefetchTrainingAssetsNode(Node):
    NODE_TYPE = "PrefetchTrainingAssets"
    CATEGORY = "Training"
    INPUTS = {"asset_refs": Port("asset_refs", ASSET_BUNDLE)}
    OUTPUTS = {"assets": Port("assets", ASSET_BUNDLE)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"assets": prefetch_training_assets(inputs["asset_refs"])} for inputs in batch]


class PrefetchOodTextSetsNode(Node):
    NODE_TYPE = "PrefetchOodTextSets"
    CATEGORY = "Training"
    INPUTS = {"ood_text_set_refs": Port("ood_text_set_refs", JSON)}
    OUTPUTS = {"ood_text_sets": Port("ood_text_sets", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"ood_text_sets": _prefetch_ood_text_sets(inputs["ood_text_set_refs"])} for inputs in batch]


def _resolve_ood_text_sets(sets: list[OodTextSet]) -> dict:
    if not sets:
        raise ValueError("SelectOodTextSets requires at least one text set")
    rows = []
    with database_session() as session:
        for item in sets:
            file_id = UUID(item.id)
            extra_file = asset_crud.get_extra_file(session, file_id)
            path = asset_crud.get_extra_file_path(session, file_id)
            if not path.is_file():
                raise ValueError(f"OOD text set file is missing: {file_id}")
            rows.append({
                "id": str(file_id),
                "name": item.name or extra_file.name,
                "line_count": item.line_count,
                "path": str(path),
                "type": extra_file.type_,
                "content_hash": extra_file.content_hash,
            })
    return {"sets": rows, "paths": [row["path"] for row in rows]}


def _prefetch_ood_text_sets(value: dict) -> dict:
    paths = [Path(str(path)) for path in value["paths"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"OOD text set files are missing: {missing}")
    return value
