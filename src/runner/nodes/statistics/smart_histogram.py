from bisect import bisect_right
from math import ceil, floor, isfinite
from typing import Any

MAX_SMART_BINS = 200


def histogram_counts(values: list[float], bins: int, range_: tuple[float, float] | None = None) -> dict[str, Any]:
    assert 0 < bins <= MAX_SMART_BINS, f"histogram bins must be between 1 and {MAX_SMART_BINS}"
    finite_values = [float(value) for value in values if isfinite(float(value))]
    lo, hi = _value_range(finite_values, range_)
    included = [value for value in finite_values if lo <= value <= hi]
    edges = _smart_edges(included, bins, lo, hi, range_ is None)
    return {"edges": edges, "counts": _count_values(included, edges)}


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _freedman_diaconis_width(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    interquartile_range = _quantile(ordered, 0.75) - _quantile(ordered, 0.25)
    if interquartile_range == 0.0:
        return None
    return 2.0 * interquartile_range / len(ordered) ** (1.0 / 3.0)


def _smart_edges(values: list[float], bins: int, lo: float, hi: float, auto_range: bool) -> list[float]:
    width = _freedman_diaconis_width(values)
    if auto_range and width is not None and width <= 1.0 and all(value.is_integer() for value in values):
        integer_slots = ceil(hi) - floor(lo) + 1
        if integer_slots <= MAX_SMART_BINS:
            start = floor(lo) - 0.5
            return [start + index for index in range(integer_slots + 1)]
    derived_bins = bins if width is None else max(bins, ceil((hi - lo) / width))
    bin_count = min(derived_bins, MAX_SMART_BINS)
    edges = [lo + ((hi - lo) * index / bin_count) for index in range(bin_count + 1)]
    edges[-1] = hi
    return edges


def _value_range(values: list[float], value_range: tuple[float, float] | None) -> tuple[float, float]:
    if value_range is not None:
        lo, hi = value_range
    elif values:
        lo, hi = min(values), max(values)
    else:
        return 0.0, 1.0
    if hi > lo:
        return lo, hi
    epsilon = 1e-9 if lo == 0 else abs(lo) * 1e-9
    return lo - epsilon, hi + epsilon


def _count_values(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        slot = len(counts) - 1 if value == edges[-1] else bisect_right(edges, value) - 1
        counts[slot] += 1
    return counts
