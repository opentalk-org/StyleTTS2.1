from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JSON
from runner.nodes.synthesis.style_reference import audio_file_style_reference
from runner.nodes.text_processing import PhonemizeSettings
from runner.nodes.text.runtime.phonemize import phonemize_text


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


class StyleReferenceSweepSettings(StrictSettings):
    voices: list[UUID] = Field(default_factory=list, title="Style references")
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


class TestingInputPayloadNode(Node):
    CATEGORY = "Testing / Inputs"
    INPUTS = {"run": Port("run", JSON)}
    OUTPUT_FIELD = "input"
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [{self.OUTPUT_FIELD: {"node_type": self.NODE_TYPE, "run": inputs["run"], "settings": self.params}} for inputs in batch]


class TestingTextPromptNode(TestingInputPayloadNode):
    NODE_TYPE = "TestingTextPrompt"
    SETTINGS = TestingTextPromptSettings
    OUTPUT_FIELD = "prompt_text"
    OUTPUTS = {"prompt_text": Port("prompt_text", JSON)}


class SelectStyleReferenceNode(TestingInputPayloadNode):
    NODE_TYPE = "SelectStyleReference"
    SETTINGS = SelectStyleReferenceSettings
    OUTPUT_FIELD = "style_reference"
    OUTPUTS = {"style_reference": Port("style_reference", JSON)}

    async def execute(self, batch, context):
        if not self.settings.reference_id:
            raise ValueError("SelectStyleReference requires reference_id")
        return [
            {
                "style_reference": _selected_style_reference(self.settings.reference_id, self.settings.style_mix, self.settings.prosody_mix)
            }
            for inputs in batch
        ]


class StyleReferenceSweepNode(TestingInputPayloadNode):
    NODE_TYPE = "StyleReferenceSweep"
    SETTINGS = StyleReferenceSweepSettings
    OUTPUT_FIELD = "style_reference"
    OUTPUTS = {"style_reference": Port("style_reference", JSON)}

    async def execute(self, batch, context):
        if not self.settings.voices:
            raise ValueError("StyleReferenceSweep requires at least one style reference audio id")
        return [
            {
                "style_reference": {
                    "kind": "style_reference_batch",
                    "references": [
                        audio_file_style_reference(reference_id)
                        for reference_id in self.settings.voices
                    ],
                    "samples_per_voice": self.settings.samples_per_voice,
                    "source": {"node_type": self.NODE_TYPE, "run": inputs["run"]},
                }
            }
            for inputs in batch
        ]


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
        return [
            {"phonemes": testing_phoneme_payload(inputs["prompt_text"], inputs["phoneme_alphabet"])}
            for inputs in batch
        ]


def testing_phoneme_payload(prompt_text: dict[str, Any], phoneme_alphabet: dict[str, Any]) -> dict[str, Any]:
    text = _prompt_setting(prompt_text, "text")
    language = _prompt_setting(prompt_text, "language")
    symbols = _alphabet_symbols(phoneme_alphabet)
    settings = PhonemizeSettings(language=language)
    phonemes = _filtered_phonemes(_testing_phonemes(text, settings), symbols)
    return {
        "kind": "phonemes",
        "text": text,
        "language": language,
        "phonemes": " ".join(phonemes),
        "phoneme_list": phonemes,
        "symbols": symbols,
        "alphabet": phoneme_alphabet,
        "source": "espeak_align",
    }


def _selected_style_reference(reference_id: str, style_mix: float, prosody_mix: float) -> dict[str, object]:
    payload = audio_file_style_reference(UUID(reference_id))
    return {**payload, "style_mix": style_mix, "prosody_mix": prosody_mix}


def _prompt_setting(prompt_text: dict[str, Any], name: str) -> str:
    if name in prompt_text:
        return str(prompt_text[name])
    return str(prompt_text["settings"][name])


def _alphabet_symbols(phoneme_alphabet: dict[str, Any]) -> list[str]:
    if "symbols" in phoneme_alphabet:
        symbols = phoneme_alphabet["symbols"]
    else:
        symbols = phoneme_alphabet["settings"]["symbols"]
    if isinstance(symbols, str):
        return symbols.split()
    return [str(symbol) for symbol in symbols]


def _testing_phonemes(text: str, settings: PhonemizeSettings) -> str:
    return phonemize_text(
        text,
        language=settings.language,
        tie=settings.tie,
        punctuation_marks=settings.punctuation_marks,
        espeak_workers=settings.espeak_workers,
        align_threads=settings.align_threads,
    )


def _filtered_phonemes(text: str, symbols: list[str]) -> list[str]:
    allowed = set(symbols)
    return [unit for unit in text if unit and not unit.isspace() and (not allowed or unit in allowed)]
