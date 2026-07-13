from bisect import bisect_right
from math import ceil, floor, isfinite, sqrt
from typing import Any

LOW_PERCENTILE = 0.005
HIGH_PERCENTILE = 0.995
MIN_PERCENTILE_VALUES = 100


def histogram_counts(values: list[float], bins: int, range_: tuple[float, float] | None = None) -> dict[str, Any]:
    assert bins > 0, "histogram bins must be positive"
    finite_values = [float(value) for value in values if isfinite(float(value))]
    if range_ is not None:
        lo, hi = _value_range(finite_values, range_)
        included = [value for value in finite_values if lo <= value <= hi]
        edges = _equal_edges(lo, hi, bins)
        return {"edges": edges, "counts": _count_values(included, edges), "underflow": 0, "overflow": 0}
    if not finite_values or len(set(finite_values)) == 1:
        lo, hi = _value_range(finite_values, None)
        edges = _equal_edges(lo, hi, 1)
        return {"edges": edges, "counts": _count_values(finite_values, edges), "underflow": 0, "overflow": 0}
    if len(finite_values) < MIN_PERCENTILE_VALUES:
        lo, hi = _value_range(finite_values, None)
        small_bin_count = min(bins, ceil(sqrt(len(finite_values))))
        edges = _equal_edges(lo, hi, small_bin_count)
        return {"edges": edges, "counts": _count_values(finite_values, edges), "underflow": 0, "overflow": 0}
    edges = _percentile_edges(finite_values, bins)
    underflow = sum(value < edges[0] for value in finite_values)
    overflow = sum(value >= edges[-1] for value in finite_values)
    included = [value for value in finite_values if edges[0] <= value < edges[-1]]
    return {"edges": edges, "counts": _count_values(included, edges), "underflow": underflow, "overflow": overflow}


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _percentile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    lo = _quantile(ordered, LOW_PERCENTILE)
    hi = _quantile(ordered, HIGH_PERCENTILE)
    if all(value.is_integer() for value in ordered):
        lo = floor(lo) - 0.5
        hi = ceil(hi) + 0.5
        integer_slots = int(hi - lo)
        if integer_slots <= bins:
            return [lo + index for index in range(integer_slots + 1)]
    return _equal_edges(lo, hi, bins)


def _equal_edges(lo: float, hi: float, bins: int) -> list[float]:
    edges = [lo + ((hi - lo) * index / bins) for index in range(bins + 1)]
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
