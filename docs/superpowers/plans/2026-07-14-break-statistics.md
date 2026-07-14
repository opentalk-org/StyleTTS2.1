# Break Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add histograms for `<break t=N>` count per audio file and pooled break duration values to dataset statistics and the statistics dashboard.

**Architecture:** Parse exact break tags from canonical segment transcript text in a focused aggregation helper. `AggregateDatasetStatistics` adds the two resulting histograms to payload version 19, and the frontend presents them as a dedicated audio histogram group.

**Tech Stack:** Python 3.12, Runflow runner nodes, React, TypeScript, Vite.

## Global Constraints

- Count only exact transcript tags shaped like `<break t=200>`.
- Include one break-count value for every audio file, including zero.
- Pool every valid `t` value as milliseconds.
- Support database-only and acoustic statistics modes without decoding extra audio.
- Do not modify stored segment schemas, break insertion, deduplication, or histogram binning.
- Run every project command through `nix develop --command`.
- Keep verification tests temporary and remove them before completion.

---

### Task 1: Break Histogram Aggregation

**Files:**
- Modify: `src/runner/nodes/statistics/aggregate_helpers.py`
- Test: `/tmp/test_break_statistics.py`

**Interfaces:**
- Consumes: audio file IDs, canonical segment records containing `source_audio_id` and `text`, histogram bin count.
- Produces: `break_histograms(file_ids: list[str], segments: list[dict[str, Any]], bins: int) -> dict[str, dict[str, Any]]`.

- [ ] **Step 1: Write the failing temporary regression test**

Create `/tmp/test_break_statistics.py` with fixtures for three files. Include valid tags at 200, 450, and 200 ms; a file without tags; and malformed `<break t=x>` and `<break 300>` strings. Assert that the returned payload contains `break_count_per_file_histogram` and `break_duration_ms_histogram`, that the count histogram contains three observations including one zero, and that the duration histogram contains exactly three observations.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
nix develop --command python /tmp/test_break_statistics.py
```

Expected: import failure because `break_histograms` does not exist.

- [ ] **Step 3: Implement exact parsing and aggregation**

In `aggregate_helpers.py`, add a compiled exact-tag expression and implement `break_histograms`. Initialize every supplied file ID to zero, parse each segment's `text`, increment its parent file count, pool integer duration values, and return both histograms through the existing `histogram_counts` helper.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run:

```bash
nix develop --command python /tmp/test_break_statistics.py
```

Expected: exit 0 with all assertions passing.

---

### Task 2: Statistics Payload Integration

**Files:**
- Modify: `src/runner/nodes/statistics/aggregate.py`
- Test: `/tmp/test_break_statistics.py`

**Interfaces:**
- Consumes: `break_histograms(file_ids, segment_records, settings.histogram_bins)` from Task 1.
- Produces: required version-19 fields `break_count_per_file_histogram` and `break_duration_ms_histogram` in the aggregate statistics payload.

- [ ] **Step 1: Extend the temporary test with payload-contract checks**

Add minimal database-mode feature records for the three file IDs and call `aggregate_dataset_statistics` with the same segment fixtures. Assert payload version 19 and equality between its two break histogram fields and the direct `break_histograms` result. The feature fixture must include every required aggregate field with empty acoustic arrays, `acoustic_metrics_available=False`, `computation_mode="database"`, `sample_selection="all"`, and `sample_requested_count=None`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
nix develop --command python /tmp/test_break_statistics.py
```

Expected: assertion failure because the payload is version 18 and lacks the helper integration.

- [ ] **Step 3: Integrate the histograms**

Import `break_histograms` in `aggregate.py`, compute the two fields from `file_ids` and `segment_records`, bump `version` from 18 to 19, and merge the returned fields into the payload.

- [ ] **Step 4: Run the regression and compilation checks**

Run:

```bash
nix develop --command python /tmp/test_break_statistics.py
nix develop --command python -m compileall -q src/runner/nodes/statistics/aggregate.py src/runner/nodes/statistics/aggregate_helpers.py
```

Expected: both commands exit 0.

---

### Task 3: Statistics Dashboard Histograms

**Files:**
- Modify: `src/frontend/src/features/statistics/api.ts`
- Modify: `src/frontend/src/features/statistics/logic.ts`

**Interfaces:**
- Consumes: version-19 histogram fields from Task 2.
- Produces: a `Break annotations` `HistogramGroup` containing `Breaks per file` and `Break duration` charts.

- [ ] **Step 1: Add the payload types**

Add both required `Histogram` properties to `StatisticsPayload` in `api.ts`.

- [ ] **Step 2: Add the histogram group**

In `audioHistogramGroups`, append a group with key `breaks`, title `Break annotations`, a caption explaining transcript pause tags, and two amber charts:

```ts
histConfig("Breaks per file", "breaks", p.break_count_per_file_histogram, "amber", "files")
histConfig("Break duration", "ms", p.break_duration_ms_histogram, "amber")
```

The group must be added before the early return for database-only reports so both computation modes display it.

- [ ] **Step 3: Run the frontend build**

Run:

```bash
nix develop --command npm run build
```

Working directory: `src/frontend`.

Expected: TypeScript and Vite build exit 0.

---

### Task 4: Registered Graph and Cleanup Verification

**Files:**
- Test: `/tmp/break_statistics_graph.json`

**Interfaces:**
- Consumes: registered `AudioSource`, `LoadAudioSegments`, `DatabaseStatisticsFeatures`, `AggregateDatasetStatistics`, and `SaveStatisticsEntry` nodes.
- Produces: temporary persisted evidence that a real database statistics graph processes a selected audio item through aggregation; the entry is deleted after inspection.

- [ ] **Step 1: Restart the managed development stack**

Run:

```bash
nix develop --command runflow-dev-stop
nix develop --command runflow-dev-session
```

Use only the shared `runflow-dev` Zellij session.

- [ ] **Step 2: Submit a registered graph with a temporary statistics entry**

Select a live audio file ID containing break tags from `/audio-files`, create `/tmp/break_statistics_graph.json`, and submit `AudioSource -> LoadAudioSegments -> DatabaseStatisticsFeatures -> AggregateDatasetStatistics -> SaveStatisticsEntry` through `POST /graphs/runs`. Give the saved entry a unique temporary name.

- [ ] **Step 3: Verify graph execution**

Use the CLI and run snapshot to assert `succeeded`, zero errors, one completed item, and one completed task for every node. Fetch the temporary entry from `/statistics`, assert payload version 19, one observation in the per-file break-count histogram, and at least one duration observation, then delete the temporary entry through `DELETE /statistics/{id}`.

- [ ] **Step 4: Run final checks and cleanup**

Run:

```bash
git diff --check
nix develop --command python /tmp/test_break_statistics.py
nix develop --command python -m compileall -q src/runner/nodes/statistics/aggregate.py src/runner/nodes/statistics/aggregate_helpers.py
nix develop --command npm run build
```

Remove `/tmp/test_break_statistics.py` and `/tmp/break_statistics_graph.json` with `apply_patch`, then confirm neither exists. Verify touched files remain under 300 lines and the statistics folders remain under 16 files.
