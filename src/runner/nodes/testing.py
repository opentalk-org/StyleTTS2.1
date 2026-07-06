from __future__ import annotations

from enum import Enum

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.core.types import UnionDataType
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import CHECKPOINT_REF, JSON


CHECKPOINT_REF_OR_JSON = UnionDataType("CHECKPOINT_REF_OR_JSON", (CHECKPOINT_REF, JSON), "Checkpoint reference or scaffold JSON")


class TestingLanguage(str, Enum):
    EN_US = "en-us"
    EN_GB = "en-gb"
    ES = "es"
    DE = "de"


class TestingTextPromptSettings(StrictSettings):
    text: str = Field(default="", title="Text")
    language: TestingLanguage = Field(default=TestingLanguage.EN_US, title="Language")


class SelectStyleReferenceSettings(StrictSettings):
    reference_id: str = Field(default="", title="Reference")
    style_mix: float = Field(default=0.7, title="Style mix", ge=0, le=1)
    prosody_mix: float = Field(default=0.5, title="Prosody mix", ge=0, le=1)


class StyleTtsSynthesisSettings(StrictSettings):
    weights_file: str = Field(default="", title="Weights file")
    diffusion_steps: int = Field(default=5, title="Diffusion steps", ge=1, le=20)
    embedding_scale: float = Field(default=1.0, title="Embedding scale", ge=0.5, le=3)


class StyleReferenceSweepSettings(StrictSettings):
    voices: list[str] = Field(default_factory=list, title="Voices")
    samples_per_voice: int = Field(default=2, title="Samples per voice", ge=1, le=5)


class TestingRunInputNode(Node):
    NODE_TYPE = "TestingRunInput"
    CATEGORY = "Testing / Inputs"
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


class MockTestingInputNode(Node):
    CATEGORY = "Testing / Inputs"
    INPUTS = {"run": Port("run", JSON)}
    OUTPUT_FIELD = "input"
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{self.OUTPUT_FIELD: {"node_type": self.NODE_TYPE, "run": inputs["run"], "settings": self.params}} for inputs in batch]


class TestingTextPromptNode(MockTestingInputNode):
    NODE_TYPE = "TestingTextPrompt"
    SETTINGS = TestingTextPromptSettings
    OUTPUT_FIELD = "prompt_text"
    OUTPUTS = {"prompt_text": Port("prompt_text", JSON)}


class SelectStyleReferenceNode(MockTestingInputNode):
    NODE_TYPE = "SelectStyleReference"
    SETTINGS = SelectStyleReferenceSettings
    OUTPUT_FIELD = "style_reference"
    OUTPUTS = {"style_reference": Port("style_reference", JSON)}


class StyleReferenceSweepNode(MockTestingInputNode):
    NODE_TYPE = "StyleReferenceSweep"
    SETTINGS = StyleReferenceSweepSettings
    OUTPUT_FIELD = "style_reference_batch"
    OUTPUTS = {"style_reference_batch": Port("style_reference_batch", JSON)}


class TestingPromptPhonemizerNode(Node):
    NODE_TYPE = "TestingPromptPhonemizer"
    CATEGORY = "Testing / Text"
    INPUTS = {
        "prompt_text": Port("prompt_text", JSON),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
    }
    OUTPUTS = {"phonemes": Port("phonemes", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{"phonemes": {"prompt": inputs["prompt_text"], "alphabet": inputs["phoneme_alphabet"], "source": "phonemizer"}} for inputs in batch]


class StyleTtsSynthesisNode(Node):
    NODE_TYPE = "StyleTtsSynthesis"
    CATEGORY = "Testing / Synthesis"
    SETTINGS = StyleTtsSynthesisSettings
    INPUTS = {
        "checkpoint": Port("checkpoint", CHECKPOINT_REF_OR_JSON),
        "prompt_text": Port("prompt_text", JSON),
        "phonemes": Port("phonemes", JSON),
        "style_reference": Port("style_reference", JSON),
    }
    OUTPUTS = {"audio_result": Port("audio_result", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"audio_result": {"node_type": self.NODE_TYPE, "settings": self.params, "inputs": inputs}} for inputs in batch]


class StyleTtsSweepSynthesisNode(Node):
    NODE_TYPE = "StyleTtsSweepSynthesis"
    CATEGORY = "Testing / Synthesis"
    INPUTS = {
        "checkpoint": Port("checkpoint", CHECKPOINT_REF_OR_JSON),
        "prompt_text": Port("prompt_text", JSON),
        "phonemes": Port("phonemes", JSON),
        "style_reference_batch": Port("style_reference_batch", JSON),
    }
    OUTPUTS = {"sweep_results": Port("sweep_results", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [{"sweep_results": {"node_type": self.NODE_TYPE, "inputs": inputs}} for inputs in batch]
