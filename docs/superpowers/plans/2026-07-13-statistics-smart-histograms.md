# Statistics Smart Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every statistics histogram data-adaptive bins and add duration-versus-total-word and duration-versus-total-character scatter plots.

**Architecture:** A focused runner utility computes Freedman–Diaconis bin edges and counts while preserving the existing histogram payload. The frontend projects the already persisted `[duration, rate, total]` rows into two additional scatter configurations, so no schema or database change is needed.

**Tech Stack:** Python 3, Pydantic runner nodes, React, TypeScript, Plotly, Nix development shell

## Global Constraints

- Run Python, frontend, and validation commands through `nix develop --command ...`.
- Do not add dependencies or change the persisted `{edges, counts}` histogram shape.
- Apply smart binning to every histogram produced by `AggregateDatasetStatistics`.
- Keep `src/runflow` domain-agnostic and unchanged.
- Do not leave committed tests or temporary probe files in the repository.
- Preserve unrelated working-tree changes.

---

### Task 1: Data-Adaptive Histogram Utility

**Files:**
- Create: `src/runner/nodes/statistics/smart_histogram.py`
- Modify: `src/runner/nodes/statistics/aggregate_helpers.py:1-24`
- Temporary test: `/tmp/runflow_test_smart_histogram.py`

**Interfaces:**
- Produces: `histogram_counts(values: list[float], bins: int, range_: tuple[float, float] | None = None) -> dict[str, Any]`
- Preserves: callers import `histogram_counts` from `runner.nodes.statistics.aggregate_helpers`
- Depends on: Python standard library only

- [ ] **Step 1: Write the failing tests**

Create `/tmp/runflow_test_smart_histogram.py` as a standard-library `unittest` module containing the three specified cases.

Use `unittest.TestCase` assertions and call `unittest.main()` under the module guard so the locked environment needs no test dependency.

- [ ] **Step 2: Run the tests and verify RED**

Run `nix develop --command python /tmp/runflow_test_smart_histogram.py`.

Expected: collection fails with `ModuleNotFoundError: No module named 'runner.nodes.statistics.smart_histogram'`.

- [ ] **Step 3: Implement `smart_histogram.py`**

Start with this public boundary:

```python
from bisect import bisect_right
from math import ceil, floor, isfinite
from typing import Any

MAX_SMART_BINS = 200


def histogram_counts(values: list[float], bins: int, range_: tuple[float, float] | None = None) -> dict[str, Any]:
    assert bins > 0, "histogram bins must be positive"
    finite_values = [float(value) for value in values if isfinite(float(value))]
    lo, hi = _value_range(finite_values, range_)
    included = [value for value in finite_values if lo <= value <= hi]
    edges = _smart_edges(included, bins, lo, hi, range_ is None)
    return {"edges": edges, "counts": _count_values(included, edges)}
```

Add module-level helpers with these exact contracts:

- `_quantile(sorted_values: list[float], fraction: float) -> float` uses linear interpolation between adjacent ranks.
- `_freedman_diaconis_width(values: list[float]) -> float | None` returns `2 * IQR / n ** (1 / 3)`, or `None` for fewer than two values or zero IQR.
- `_smart_edges(values, bins, lo, hi, auto_range)` derives `max(bins, ceil((hi - lo) / width))`, capped at 200. For integer-only auto-ranged input with width at most `1.0` and at most 200 integer slots, it returns unit edges from `floor(lo) - 0.5` through `ceil(hi) + 0.5`; this intentional integer exception uses one bin per possible integer rather than padding to `bins`.
- `_value_range(values, range_)` preserves explicit bounds, returns `(0.0, 1.0)` for empty input, and expands constants with the existing relative epsilon rule.
- `_count_values(values, edges)` uses `bisect_right` and puts a value equal to the final edge into the final bin.

- [ ] **Step 4: Re-export through the existing helper boundary**

In `aggregate_helpers.py`, remove its current `histogram_counts` and `_range_for_pair`, then add:

```python
from runner.nodes.statistics.smart_histogram import histogram_counts
```

Keep `isfinite` and `_range_for`, which `pooled_histogram` still uses. This switches every existing aggregate histogram without editing `aggregate.py`.

- [ ] **Step 5: Verify GREEN and remove the temporary test**

Run:

```bash
nix develop --command python /tmp/runflow_test_smart_histogram.py
rg -n "histogram_counts\(" src/runner/nodes/statistics
rm /tmp/runflow_test_smart_histogram.py
```

Expected: `Ran 3 tests` and `OK`; all aggregate histogram call sites still resolve through `aggregate_helpers`; the temporary test is removed.

- [ ] **Step 6: Commit the isolated runner change**

```bash
git add src/runner/nodes/statistics/smart_histogram.py src/runner/nodes/statistics/aggregate_helpers.py
git commit -m "feat: add smart statistics histogram bins"
```

### Task 2: Duration-versus-Count Scatter Projections

**Files:**
- Modify: `src/frontend/src/features/statistics/logic.ts:90-104`
- Temporary test: `/tmp/runflow_test_statistics_scatter.mts`

**Interfaces:**
- Consumes: `ScatterPoint` rows shaped as `[duration, rate, total]`
- Produces: `rateScatters(payload)` with six configurations
- Preserves: two-column rows are excluded only from projections that require `total`

- [ ] **Step 1: Write the failing frontend probe**

Create `/tmp/runflow_test_statistics_scatter.mts`:

```typescript
import assert from "node:assert/strict";
import { rateScatters } from "/workspace/styletts_studio_v2/src/frontend/src/features/statistics/logic.ts";

const payload = {
  words_per_second_scatter: [[2, 3, 6], [4, 2]],
  chars_per_second_scatter: [[2, 7, 14], [4, 5]],
};
const charts = rateScatters(payload as never);

assert.equal(charts.length, 6);
assert.deepEqual(charts[4], {
  title: "Total words", unit: "vs duration", points: [[2, 6]],
  xLabel: "Duration (s)", yLabel: "Words", tone: "blue",
});
assert.deepEqual(charts[5], {
  title: "Total characters", unit: "vs duration", points: [[2, 14]],
  xLabel: "Duration (s)", yLabel: "Characters", tone: "emerald",
});
```

- [ ] **Step 2: Run the probe and verify RED**

Run `nix develop --command node --experimental-strip-types /tmp/runflow_test_statistics_scatter.mts`.

Expected: assertion failure because `charts.length` is `4`.

- [ ] **Step 3: Add the two scatter configurations**

In `rateScatters`, compute `wordsWithTotal` and `charsWithTotal` once, reuse them in the current total-axis rate plots, then append:

```typescript
{ title: "Total words", unit: "vs duration", points: wordsWithTotal.map((r) => [r[0]!, r[2]!]), xLabel: "Duration (s)", yLabel: "Words", tone: "blue" },
{ title: "Total characters", unit: "vs duration", points: charsWithTotal.map((r) => [r[0]!, r[2]!]), xLabel: "Duration (s)", yLabel: "Characters", tone: "emerald" },
```

No `StatisticsScreen.tsx` edit is needed because it already renders every returned scatter in a two-column grid.

- [ ] **Step 4: Verify GREEN and remove the probe**

```bash
nix develop --command node --experimental-strip-types /tmp/runflow_test_statistics_scatter.mts
nix develop --command npm --prefix src/frontend run build
rm /tmp/runflow_test_statistics_scatter.mts
```

Expected: the probe exits successfully; TypeScript and Vite build successfully; the probe is removed.

- [ ] **Step 5: Commit the frontend change**

```bash
git add src/frontend/src/features/statistics/logic.ts
git commit -m "feat: plot duration against word and character counts"
```

### Task 3: Integrated Verification

**Files:**
- Verify: `src/runner/nodes/statistics/smart_histogram.py`
- Verify: `src/runner/nodes/statistics/aggregate_helpers.py`
- Verify: `src/frontend/src/features/statistics/logic.ts`

**Interfaces:**
- Confirms: smart edges remain compatible with the existing variable-width Plotly bars
- Confirms: count conservation, build health, file-size limits, and clean temporary-test state

- [ ] **Step 1: Run Python and frontend checks**

```bash
nix develop --command python -m compileall -q src/runner/nodes/statistics
nix develop --command npm --prefix src/frontend run build
```

Expected: both commands exit successfully without Python, TypeScript, or Vite errors.

- [ ] **Step 2: Verify final histogram invariants**

```bash
nix develop --command python -c 'from runner.nodes.statistics.aggregate_helpers import histogram_counts; h = histogram_counts([1.0] * 30 + [2.0] * 25 + [3.0] * 20 + [4.0] * 15 + [100.0], 10); assert len(h["edges"]) == len(h["counts"]) + 1; assert sum(h["counts"]) == 91; assert len(h["counts"]) <= 200'
```

Expected: command exits successfully without output.

- [ ] **Step 3: Inspect scope and constraints**

```bash
git diff --check HEAD~2..HEAD
wc -l src/runner/nodes/statistics/smart_histogram.py src/runner/nodes/statistics/aggregate_helpers.py src/frontend/src/features/statistics/logic.ts
git status --short
```

Expected: no whitespace errors; every file is under 300 lines; no temporary tests appear; unrelated pre-existing changes remain untouched.

- [ ] **Step 4: Review the implementation commits**

```bash
git log -3 --oneline
git show --stat --oneline HEAD~1..HEAD
```

Expected: the implementation commits contain only the smart histogram utility/helper changes and the scatter configuration change.
