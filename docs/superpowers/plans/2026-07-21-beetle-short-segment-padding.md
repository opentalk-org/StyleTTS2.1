# Beetle Short-Segment Padding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Beetle's effective lower duration cutoff, enforce the configured upper cutoff, and zero-pad short Stage 1 segments to its fixed valid window.

**Architecture:** Duration eligibility is owned by `DatabaseSegmentIndex` and receives only `maximum_seconds`. Conditional cut planning receives the same maximum. Stage 1 planning emits one zero-origin window for short sources, while collation pads targets and derives `frame_mask` from real source duration.

**Tech Stack:** Python 3.12, Pydantic, PyTorch, pytest through the Nix development shell.

## Global Constraints

- Keep `src/runflow` domain-agnostic.
- Run all Python and pytest commands through `nix develop --command python -m ...`.
- Do not restart or alter the running Stage 1 process.
- Temporary regression tests must be removed before handoff.

---

### Task 1: Configured Maximum Duration

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/data.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/data/index.py`
- Modify: `src/runner/nodes/training/beetle/data/cuts.py`
- Modify: `src/runner/nodes/training/beetle/data/sampling.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/data/records.py`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Temporary test: `/tmp/test_beetle_duration_limits.py`

**Interfaces:**
- `DatabaseSegmentIndex.build(selection, languages, maximum_seconds, page_size, callbacks)`
- `DatabaseSegmentIndex.from_references(dataset_id, selected_audio_ids, languages, maximum_seconds, references)`
- `ContinuousBatchPlanner(..., maximum_seconds, grouping, shard)`
- `CutPlanner(index, maximum_seconds)`

- [ ] Write temporary failing tests proving a sub-second reference remains eligible, a reference above a custom maximum is excluded, and the default config uses a zero lower duration.
- [ ] Run `nix develop --command python -m pytest -q /tmp/test_beetle_duration_limits.py` and confirm failures reference the current lower limit and missing maximum plumbing.
- [ ] Retain the lower field for checkpoint compatibility but stop using it, thread `maximum_seconds` through the listed interfaces, replace `1–45` checks with positive-range invariants plus the configured maximum, and leave the active run YAML byte-for-byte unchanged.
- [ ] Re-run the temporary test and confirm all duration-limit cases pass.

### Task 2: Stage 1 Short-Source Padding

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/stage1_records.py`
- Modify: `src/runner/nodes/training/beetle/data/stage1_loader.py`
- Temporary test: `/tmp/test_beetle_stage1_short_padding.py`

**Interfaces:**
- `Stage1WindowGeometry.plans(item)` returns one `Stage1WindowPlan(item.key, 0, 0)` when the source contains fewer latent frames than the fixed window.
- `Stage1WindowLoader.collate(fetched)` returns fixed-size zero-padded targets and a fully valid fixed-window mask so adversarial crops may include learned trailing silence.

- [ ] Write a temporary failing test with a 0.4-second source proving planning succeeds, tensors are padded to 64 mel frames and 19,200 samples, padding is zero, and all fixed-window frames are true in `frame_mask`.
- [ ] Run `nix develop --command python -m pytest -q /tmp/test_beetle_stage1_short_padding.py` and confirm it fails at the current short-window exception.
- [ ] Change planning and collation minimally: emit one short-source plan; copy available mel/waveform prefixes into zero tensors; compute real frames from `geometry.mel_frames(item)`; preserve contextual encoder masking.
- [ ] Add an extremely short source case to ensure preprocessing/STFT minimum padding does not turn artificial frames into valid target frames.
- [ ] Re-run both temporary test files and confirm all cases pass.

### Task 3: Integrated Verification and Cleanup

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/collate.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Remove: `/tmp/test_beetle_duration_limits.py`
- Remove: `/tmp/test_beetle_stage1_short_padding.py`

**Interfaces:** None.

- [ ] Pad conditional batch tensors to at least 64 frames and let adversarial segment selection use the padded crop region for short examples without changing the synthesis frame mask.
- [ ] Run both temporary regression files together through Nix and record the passing count.
- [ ] Run `nix develop --command python -m compileall -q src/runner/nodes/training/beetle` and confirm exit code 0.
- [ ] Load both Beetle default and active Stage 1 YAML files through `load_config` and confirm `maximum_seconds == 45.0`.
- [ ] Confirm the live training PID and command are unchanged.
- [ ] Remove the temporary test files and inspect `git diff --check` plus `git status --short`.
