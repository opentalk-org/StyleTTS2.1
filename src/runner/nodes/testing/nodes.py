from __future__ import annotations

from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JSON
from runner.nodes.synthesis.style_reference import audio_file_style_reference
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.voices import crud as voice_crud


class TestingTextPromptSettings(StrictSettings):
    text: str = Field(default="", title="Text")
    language: str = Field(default="en-us", title="Language")


class SelectStyleReferenceSettings(StrictSettings):
    reference_id: str = Field(default="", title="Reference")
    alpha: float = Field(default=0.7, title="Alpha", ge=0, le=1)
    beta: float = Field(default=0.3, title="Beta", ge=0, le=1)


class StyleReferenceSweepSettings(StrictSettings):
    voices: list[UUID] = Field(default_factory=list, title="Style references")
    samples_per_voice: int = Field(default=2, title="Samples per voice", ge=1, le=5)
    alpha: float = Field(default=0.7, title="Alpha", ge=0, le=1)
    beta: float = Field(default=0.3, title="Beta", ge=0, le=1)


class TestingRunInputNode(Node):
    NODE_TYPE = "TestingRunInput"
    CATEGORY = "Testing"
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
    CATEGORY = "Testing"
    INPUTS = {"run": Port("run", JSON)}
    OUTPUT_FIELD = "input"
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        settings = self.settings.model_dump(mode="json")
        return [{self.OUTPUT_FIELD: {"node_type": self.NODE_TYPE, "run": inputs["run"], "settings": settings}} for inputs in batch]


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
                "style_reference": _selected_style_reference(self.settings.reference_id, self.settings.alpha, self.settings.beta)
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
            raise ValueError("StyleReferenceSweep requires at least one voice")
        references = _voice_style_references(self.settings.voices, self.settings.alpha, self.settings.beta)
        return [
            {
                "style_reference": {
                    "kind": "style_reference_batch",
                    "references": references,
                    "samples_per_voice": self.settings.samples_per_voice,
                    "source": {"node_type": self.NODE_TYPE, "run": inputs["run"]},
                }
            }
            for inputs in batch
        ]


def _selected_style_reference(reference_id: str, alpha: float, beta: float) -> dict[str, object]:
    payload = audio_file_style_reference(UUID(reference_id))
    return {**payload, "alpha": alpha, "beta": beta}


def _voice_style_references(voice_ids: list[UUID], alpha: float, beta: float) -> list[dict[str, object]]:
    with database_session() as session:
        audio_files = list(audio_crud.list_audio_files(session))
        voices = {voice.id: voice.name for voice in voice_crud.list_voices(session)}
        reference_ids = [_voice_reference_audio_id(audio_files, voice_id, voices.get(voice_id, "")) for voice_id in voice_ids]
    return [{**audio_file_style_reference(reference_id), "alpha": alpha, "beta": beta} for reference_id in reference_ids]


def _voice_reference_audio_id(audio_files, voice_id: UUID, voice_name: str = "") -> UUID:
    voice = str(voice_id)
    name = voice_name.strip()
    matches = [
        item
        for item in audio_files
        if str(item.metadata_.get("voice_id") or "") == voice
        or (name and str(item.metadata_.get("speaker") or "").strip() == name)
        or any(str(segment.get("voice_id") or "") == voice for segment in item.segments)
        or (name and any(str(segment.get("speaker") or "").strip() == name for segment in item.segments))
    ]
    if not matches:
        label = f"{voice_name} ({voice_id})" if voice_name else str(voice_id)
        raise KeyError(f"Voice has no audio reference: {label}")
    return max(matches, key=lambda item: item.duration).id
