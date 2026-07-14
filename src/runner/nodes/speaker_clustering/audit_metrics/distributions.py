from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import blake2b
from heapq import heapreplace, heappush
from math import isfinite
from struct import pack
from typing import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class ScoreSample:
    key: str
    value: float


class ScoreDistribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    minimum: float | None
    q05: float | None
    q25: float | None
    median: float | None
    q75: float | None
    q95: float | None
    maximum: float | None
    mean: float | None


def score_distribution(
    values: Iterable[float | ScoreSample], maximum_values: int = 100_000
) -> ScoreDistribution:
    if maximum_values <= 0:
        raise ValueError("score distribution maximum_values must be positive")
    reservoir: list[tuple[int, str, float]] = []
    occurrences: dict[float, int] = {}
    count = 0
    total = Fraction(0)
    minimum = float("inf")
    maximum = float("-inf")
    for raw_value in values:
        if isinstance(raw_value, ScoreSample):
            value = float(raw_value.value)
            key = raw_value.key
        else:
            value = float(raw_value)
            occurrence = occurrences.get(value, 0)
            occurrences[value] = occurrence + 1
            key = f"{value.hex()}:{occurrence}"
        if not isfinite(value):
            raise ValueError("audit score distribution requires finite values")
        count += 1
        total += Fraction.from_float(value)
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        digest = blake2b(key.encode("utf-8") + pack("!d", value), digest_size=16)
        priority = int.from_bytes(digest.digest())
        item = (-priority, key, value)
        if len(reservoir) < maximum_values:
            heappush(reservoir, item)
        elif priority < -reservoir[0][0]:
            heapreplace(reservoir, item)
    if count == 0:
        return _empty_distribution()
    sample = np.asarray([value for _priority, _key, value in reservoir], dtype=np.float64)
    quantiles = np.quantile(sample, [0.05, 0.25, 0.5, 0.75, 0.95])
    return ScoreDistribution(
        count=count,
        sampled_count=len(sample),
        minimum=minimum,
        q05=float(quantiles[0]),
        q25=float(quantiles[1]),
        median=float(quantiles[2]),
        q75=float(quantiles[3]),
        q95=float(quantiles[4]),
        maximum=maximum,
        mean=float(total / count),
    )


def _empty_distribution() -> ScoreDistribution:
    return ScoreDistribution(
        count=0,
        sampled_count=0,
        minimum=None,
        q05=None,
        q25=None,
        median=None,
        q75=None,
        q95=None,
        maximum=None,
        mean=None,
    )
