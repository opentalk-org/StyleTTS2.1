from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.assets.credentials import openrouter_token
from runner.nodes.datatypes import TextPort
from runner.nodes.languages import Language
from runner.nodes.text_generation.defaults import DEFAULT_KEYWORDS, DEFAULT_SEED_SENTENCES, DEFAULT_TOPICS
from runner.nodes.text_generation.openrouter import request_sentences
from runner.nodes.text_generation.sampling import LengthDistribution, LengthMode, SentenceSpec, plan_sentence_specs

_SYSTEM_PROMPT = (
    "You write natural, spoken-style sentences for a text-to-speech dataset in {language}. "
    "Every sentence must be fully normalized for speech: spell out ALL numbers, dates, times, "
    "currencies, ordinals, percentages, and abbreviations as words (for example '10.02.2026' becomes "
    "'the tenth of February twenty twenty-six', '$5' becomes 'five dollars', '%' becomes 'percent', "
    "'Dr.' becomes 'Doctor'). Use no digits and no symbols other than commas, periods, question marks, "
    "exclamation marks, and apostrophes. Each sentence must be a single, self-contained, natural "
    "utterance suitable for reading aloud. Match the requested approximate word count, weave in the "
    "given keywords naturally, reflect the given topic, and vary wording and sentence structure. "
    "Return exactly the requested number of sentences, in the same order as the requests."
)


class OpenRouterGenerateSettings(StrictSettings):
    model: str = Field(default="openai/gpt-4o-mini", title="OpenRouter model")
    count: int = Field(default=64, ge=1, le=4096, title="Number of texts")
    language: Language = Field(default=Language.ENGLISH, title="Language")
    batch_size: int = Field(default=16, ge=1, le=64, title="Texts per request")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, title="Temperature")
    seed: int = Field(default=0, title="Sampling seed")
    length_mode: LengthMode = Field(default=LengthMode.GAUSSIAN, title="Length distribution")
    length_mean_words: float = Field(default=12.0, ge=1.0, le=200.0, title="Mean words (gaussian)")
    length_std_words: float = Field(default=5.0, ge=0.0, le=100.0, title="Std words (gaussian)")
    length_min_words: int = Field(default=4, ge=1, le=200, title="Min words")
    length_max_words: int = Field(default=24, ge=1, le=400, title="Max words")
    keywords_per_text: int = Field(default=2, ge=0, le=10, title="Keywords per text")
    seed_sentences: list[str] = Field(default_factory=lambda: list(DEFAULT_SEED_SENTENCES), title="Example sentences")
    keywords: list[str] = Field(default_factory=lambda: list(DEFAULT_KEYWORDS), title="Keywords")
    topics: list[str] = Field(default_factory=lambda: list(DEFAULT_TOPICS), title="Topics")


class OpenRouterGenerateTextsNode(Node):
    NODE_TYPE = "OpenRouterGenerateTexts"
    DESCRIPTION = "Generate TTS-ready sentences with a large language model via OpenRouter, streaming out one text per item to feed synthesis. Choose the model, language, how many texts to produce, and length distribution, then guide style with example sentences, keywords, and topics; every sentence is fully normalized for speech (numbers, dates, and symbols spelled out). Sampling is deterministic from the seed, so a run is reproducible. Requires an OpenRouter API key."
    CATEGORY = "Text"
    SETTINGS = OpenRouterGenerateSettings
    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    OUTPUTS = {"text": TextPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._specs = self._plan_specs()
        self._cursor = 0

    def remaining_items(self, context) -> int:
        return len(self._specs) - self._cursor

    async def execute(self, batch, context):
        token = openrouter_token()
        if not token:
            raise ValueError("openrouter_token_missing: set it in Settings > OpenRouter or OPENROUTER_API_KEY")
        specs = self._specs[self._cursor : self._cursor + self.settings.batch_size]
        context.check_cancel()
        texts = await asyncio.to_thread(self._generate, token, specs)
        self._cursor += len(specs)
        await context.report_progress(self.id, self._cursor, len(self._specs), f"generated {self._cursor}/{len(self._specs)} texts")
        return [{"text": text} for text in texts]

    def _generate(self, token: str, specs: list[SentenceSpec]) -> list[str]:
        language = self.settings.language.display_name
        system_prompt = _SYSTEM_PROMPT.format(language=language)
        user_prompt = _build_user_prompt(specs, language)
        texts = request_sentences(
            token=token,
            model=self.settings.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.settings.temperature,
        )
        if len(texts) != len(specs):
            texts = request_sentences(
                token=token,
                model=self.settings.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.settings.temperature,
            )
        if not texts:
            raise RuntimeError("openrouter_returned_no_sentences")
        return texts[: len(specs)]

    def _plan_specs(self) -> list[SentenceSpec]:
        distribution = LengthDistribution(
            mode=self.settings.length_mode,
            mean_words=self.settings.length_mean_words,
            std_words=self.settings.length_std_words,
            min_words=self.settings.length_min_words,
            max_words=self.settings.length_max_words,
        )
        return plan_sentence_specs(
            count=self.settings.count,
            distribution=distribution,
            seed_sentences=self.settings.seed_sentences,
            keywords=self.settings.keywords,
            topics=self.settings.topics,
            keywords_per_text=self.settings.keywords_per_text,
            seed=self.settings.seed,
        )


def _build_user_prompt(specs: list[SentenceSpec], language: str) -> str:
    requests = [
        {
            "index": spec.index,
            "target_words": spec.target_words,
            "topic": spec.topic,
            "keywords": list(spec.keywords),
            "style_example": spec.example,
        }
        for spec in specs
    ]
    header = (
        f"Write {len(specs)} sentences in {language}. For each request, produce one sentence of about "
        "'target_words' words, on the given 'topic', naturally including every keyword in 'keywords', "
        "in the natural style of 'style_example' (do not copy it). Keep the same order.\n\nRequests:\n"
    )
    return header + json.dumps(requests, ensure_ascii=False, indent=2)
