# Readable Statistics Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected dynamic-bin histograms with fixed, readable core bins and explicit tail bars while preserving the two duration-versus-count scatter plots.

**Architecture:** The runner computes a 0.5th–99.5th percentile display range for automatically ranged datasets of at least 100 values, returns fixed core bins plus underflow and overflow counts, and keeps explicit semantic ranges unchanged. The Plotly component renders tail counts as adjacent labeled bars, so outliers remain visible without compressing the dense region.

**Tech Stack:** Python 3 standard library, React, TypeScript, Plotly, Nix development shell

## Global Constraints

- Run Python and frontend commands through `nix develop --command ...`.
- Do not add dependencies or modify `src/runflow`.
- Use the configured bin count; never expand to 200 bins dynamically.
- Preserve total finite counts across core, underflow, and overflow for auto-ranged histograms.
- Keep explicit-range inclusion behavior unchanged.
- Remove temporary tests before completion and preserve unrelated working-tree changes.

---

### Task 1: Fixed Core Bins and Tail Counts

**Files:**
- Modify: `src/runner/nodes/statistics/smart_histogram.py`
- Temporary test: `/tmp/runflow_test_readable_histogram.py`

**Interfaces:**
- Produces: `histogram_counts(...) -> {"edges": list[float], "counts": list[int], "underflow": int, "overflow": int}`
- Preserves: imports through `runner.nodes.statistics.aggregate_helpers.histogram_counts`

- [ ] **Step 1: Write the failing regression test**

Create a standard-library `unittest` module with this core case:

```python
class ReadableHistogramTests(unittest.TestCase):
    def test_long_tail_focuses_integer_core_and_reports_overflow(self):
        values = [1.0] * 8500 + [2.0] * 80 + [3.0] * 45 + [4.0] * 25 + list(range(5, 46))
        result = histogram_counts(values, bins=50)
        self.assertEqual(result["edges"][:5], [0.5, 1.5, 2.5, 3.5, 4.5])
        self.assertLessEqual(len(result["counts"]), 50)
        self.assertGreater(result["overflow"], 0)
        self.assertEqual(sum(result["counts"]) + result["underflow"] + result["overflow"], len(values))

    def test_explicit_range_keeps_fixed_bins_without_tail_buckets(self):
        result = histogram_counts([-1.0, 0.0, 0.5, 1.0, 2.0], bins=10, range_=(0.0, 1.0))
        self.assertEqual(len(result["counts"]), 10)
        self.assertEqual(result["underflow"], 0)
        self.assertEqual(result["overflow"], 0)
        self.assertEqual(sum(result["counts"]), 3)
```

Include the existing empty and constant assertions and `unittest.main()`.

- [ ] **Step 2: Verify RED**

Run `nix develop --command python /tmp/runflow_test_readable_histogram.py`.

Expected: failures because the current implementation returns no tail fields and expands the long tail dynamically.

- [ ] **Step 3: Replace dynamic bin selection**

In `smart_histogram.py`:

```python
LOW_PERCENTILE = 0.005
HIGH_PERCENTILE = 0.995
MIN_PERCENTILE_VALUES = 100
```

- Remove `MAX_SMART_BINS` and `_freedman_diaconis_width`.
- Validate `bins > 0`.
- For auto ranges with at least 100 nonconstant values, compute percentile boundaries with `_quantile`.
- For integer-only data, round the lower boundary down and upper boundary up to half-integer edges. If the integer span is at most `bins`, emit one unit-width bin per integer.
- Otherwise emit exactly `bins` equal-width core bins.
- Count auto-range values below the first edge as `underflow`, values at or above the final edge as `overflow`, and remaining values in ordinary bins.
- For explicit ranges, retain the current inclusive final edge and skip out-of-range values; return zero tail counts.
- For empty, constant, or fewer-than-100 values, use the complete deterministic range and zero tail counts.

- [ ] **Step 4: Verify GREEN and remove the test**

```bash
nix develop --command python /tmp/runflow_test_readable_histogram.py
rm /tmp/runflow_test_readable_histogram.py
```

Expected: all tests report `OK`.

- [ ] **Step 5: Commit the runner correction**

```bash
git add src/runner/nodes/statistics/smart_histogram.py
git commit -m "fix: focus statistics histograms on readable ranges"
```

### Task 2: Tail Bars in Plotly Histograms

**Files:**
- Modify: `src/frontend/src/features/statistics/api.ts`
- Modify: `src/frontend/src/features/statistics/logic.ts`
- Create: `src/frontend/src/features/statistics/charts/histogramGeometry.ts`
- Modify: `src/frontend/src/features/statistics/charts/Histogram.tsx`
- Modify: `src/frontend/src/features/statistics/StatisticsScreen.tsx`
- Temporary test: `/tmp/runflow_test_histogram_bars.mts`

**Interfaces:**
- Consumes: histogram `underflow` and `overflow` counts
- Produces: pure `histogramBars(edges, counts, underflow, overflow)` arrays used by `Histogram`

- [ ] **Step 1: Write a failing bar-geometry probe**

Create `/tmp/runflow_test_histogram_bars.mts` that imports `histogramBars` from `histogramGeometry.ts` and asserts:

```typescript
assert.deepEqual(histogramBars([0.5, 1.5, 2.5], [80, 15], 0, 5), {
  centers: [1, 2, 3],
  widths: [1, 1, 1],
  counts: [80, 15, 5],
  ranges: ["0.5 – 1.5", "1.5 – 2.5", "≥ 2.5"],
});
```

- [ ] **Step 2: Verify RED**

Run `nix develop --command node --experimental-strip-types /tmp/runflow_test_histogram_bars.mts`.

Expected: import failure because `histogramBars` does not exist.

- [ ] **Step 3: Implement and wire tail geometry**

- Extend `Histogram` in `api.ts` with optional `underflow?: number` and `overflow?: number` for already persisted entries.
- Extend `HistogramConfig` and `histConfig` in `logic.ts` with normalized numeric tail counts.
- Implement `histogramBars` in the pure `histogramGeometry.ts` module. Prepend a same-width bar before the first core bar when underflow is positive and append one after the last core bar when overflow is positive. Label them `< first edge` and `≥ final edge`.
- Make `Histogram` render the returned arrays.
- Pass tail counts through audio, voice, and corpus histogram calls in `StatisticsScreen.tsx`; extend `CorpusData` with the two length-tail values.

- [ ] **Step 4: Verify GREEN and build**

```bash
nix develop --command node --experimental-strip-types /tmp/runflow_test_histogram_bars.mts
nix develop --command npm --prefix src/frontend run build
rm /tmp/runflow_test_histogram_bars.mts
```

Expected: the probe exits successfully and Vite reports `built`.

- [ ] **Step 5: Commit the renderer correction**

```bash
git add src/frontend/src/features/statistics/api.ts src/frontend/src/features/statistics/logic.ts src/frontend/src/features/statistics/charts/histogramGeometry.ts src/frontend/src/features/statistics/charts/Histogram.tsx src/frontend/src/features/statistics/StatisticsScreen.tsx
git commit -m "fix: render histogram tails without compressing the core"
```

### Task 3: Real-Graph and Visual-Data Verification

**Files:**
- Verify the files from Tasks 1 and 2

- [ ] **Step 1: Run static verification**

```bash
nix develop --command python -m compileall -q src/runner/nodes/statistics
nix develop --command npm --prefix src/frontend run build
git diff --check HEAD~2..HEAD
```

- [ ] **Step 2: Run a real database statistics graph**

Submit selected audio IDs through `POST /graphs/runs` using `AudioSource → LoadAudioSegments → DatabaseStatisticsFeatures → AggregateDatasetStatistics → SaveStatisticsEntry`. Assign the response's `.run_id` value to the shell variable `run_id`, then inspect it with:

```bash
nix develop --command python -m cli run "$run_id"
nix develop --command python -m cli logs "$run_id"
```

Expected: `succeeded`, with all five nodes completing normally.

- [ ] **Step 3: Inspect a 10,000-file payload shape**

Recompute or inspect the available dataset entry and verify that voice histograms use no more than 50 core bins, expose nonzero tail counts where needed, and conserve `sum(counts) + underflow + overflow`.

- [ ] **Step 4: Remove the temporary saved statistics entry and inspect scope**

Delete the verification statistics entry through its API, confirm every touched source file is under 300 lines, and ensure unrelated dirty files remain untouched.
