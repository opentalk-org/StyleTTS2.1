from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic import ConfigDict, Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JsonPort
from runner.nodes.tts.piper_catalog import (
    PiperSelection,
    fetch_piper_catalog,
    select_voices,
)
from runner.nodes.tts.piper_download import (
    download_piper_voice,
    resolve_downloaded_piper_voice,
)
from runner.nodes.tts.voices import (
    PiperVoiceModel,
    TtsEngine,
    piper_voice_payload,
    voice_batch_payload,
)


class PiperSelectionMode(StrEnum):
    EXPLICIT = "explicit"
    ALL_MATCHING = "all_matching"
    RANDOM = "random"


class PiperVoiceSelectionSettings(StrictSettings):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"x-piper-catalog-url": "/piper/voices"},
    )

    mode: PiperSelectionMode = Field(
        default=PiperSelectionMode.EXPLICIT, title="Selection mode"
    )
    voice_ids: list[str] = Field(default_factory=list, title="Piper voices")
    languages: list[str] = Field(default_factory=list, title="Language families")
    locales: list[str] = Field(default_factory=list, title="Locales")
    qualities: list[str] = Field(default_factory=list, title="Qualities")
    count: int = Field(default=1, ge=1, title="Random voice count")
    seed: int = Field(default=0, title="Seed")
    samples_per_voice: int = Field(default=1, ge=1, le=16, title="Samples per voice")
    download_missing: bool = Field(default=True, title="Download missing voices")


class PiperVoiceSelectionNode(Node):
    NODE_TYPE = "PiperVoiceSelection"
    DESCRIPTION = "Select Piper models as voices by ID or language/locale/quality filters. Optionally downloads missing model/config pairs and emits the standard TTS voice or voice-batch payload for Piper synthesis."
    CATEGORY = "TTS"
    SETTINGS = PiperVoiceSelectionSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"voice": JsonPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context) -> int:
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"voice node already emitted: {self.id}"
        self._emitted = True
        payload = await asyncio.to_thread(self._select_payload, context.check_cancel)
        return [{"voice": payload}]

    def _select_payload(self, check_cancel):
        catalog = fetch_piper_catalog()
        if (
            self.settings.mode is PiperSelectionMode.EXPLICIT
            and not self.settings.voice_ids
        ):
            raise ValueError("piper_explicit_selection_requires_voice_ids")
        count = (
            self.settings.count
            if self.settings.mode is PiperSelectionMode.RANDOM
            else None
        )
        voice_ids = (
            tuple(self.settings.voice_ids)
            if self.settings.mode is PiperSelectionMode.EXPLICIT
            else ()
        )
        selected = select_voices(
            catalog,
            PiperSelection(
                voice_ids=voice_ids,
                languages=tuple(self.settings.languages),
                locales=tuple(self.settings.locales),
                qualities=tuple(self.settings.qualities),
                count=count,
                seed=self.settings.seed,
            ),
        )
        payloads = []
        for voice in selected:
            check_cancel()
            checkpoint = (
                download_piper_voice(voice)
                if self.settings.download_missing
                else resolve_downloaded_piper_voice(voice.voice_id)
            )
            metadata = checkpoint.metadata["metadata"]
            payloads.append(
                piper_voice_payload(
                    PiperVoiceModel(
                        voice_id=voice.voice_id,
                        checkpoint_id=str(checkpoint.checkpoint_id),
                        language=voice.language.family,
                        locale=voice.language.code,
                        quality=voice.quality,
                        sample_rate=int(metadata["sample_rate"]),
                    )
                )
            )
        if len(payloads) == 1 and self.settings.samples_per_voice == 1:
            return payloads[0]
        return voice_batch_payload(
            TtsEngine.PIPER, payloads, self.settings.samples_per_voice
        )
