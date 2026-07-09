from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class LengthMode(str, Enum):
    """How a target word count is drawn for each generated sentence."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class LengthDistribution:
    mode: LengthMode
    mean_words: float
    std_words: float
    min_words: int
    max_words: int

    def sample(self, rng: random.Random) -> int:
        if self.mode == LengthMode.UNIFORM:
            return rng.randint(self.min_words, self.max_words)
        drawn = rng.gauss(self.mean_words, self.std_words)
        return int(round(min(self.max_words, max(self.min_words, drawn))))


@dataclass(frozen=True)
class SentenceSpec:
    """A single sentence request: what the model should write and how long."""

    index: int
    target_words: int
    topic: str
    example: str
    keywords: tuple[str, ...]


def plan_sentence_specs(
    *,
    count: int,
    distribution: LengthDistribution,
    seed_sentences: list[str],
    keywords: list[str],
    topics: list[str],
    keywords_per_text: int,
    seed: int,
) -> list[SentenceSpec]:
    """Deterministically plan every sentence to generate, before any network call.

    Sampling is seeded so a run is reproducible and ``remaining_items`` can be
    reported up front without contacting OpenRouter.
    """
    if not seed_sentences:
        raise ValueError("openrouter_generate_requires_seed_sentences")
    rng = random.Random(seed)
    specs: list[SentenceSpec] = []
    for index in range(count):
        target = distribution.sample(rng)
        topic = rng.choice(topics) if topics else ""
        example = rng.choice(seed_sentences)
        picked = _sample_keywords(rng, keywords, keywords_per_text)
        specs.append(SentenceSpec(index=index, target_words=target, topic=topic, example=example, keywords=picked))
    return specs


def _sample_keywords(rng: random.Random, keywords: list[str], count: int) -> tuple[str, ...]:
    if count <= 0 or not keywords:
        return ()
    take = min(count, len(keywords))
    return tuple(rng.sample(keywords, take))
