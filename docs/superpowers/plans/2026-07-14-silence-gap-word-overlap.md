# Silence Gap and Word Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge short non-silent gaps inside pauses and reject break candidates whose union overlap with aligned words exceeds a configurable ratio.

**Architecture:** Extend audio-level silence merging with a millisecond gap allowance, then apply segment-level word-overlap rejection before boundary selection. Keep both policies in their existing focused modules and expose exact runtime defaults through `InsertSilenceBreaksSettings`.

**Tech Stack:** Python 3.12, NumPy, SoundFile, Pydantic, runflow nodes, Nix development shell, runflow graph API and CLI.

## Global Constraints

- Run Python, backend, runner, and CLI commands through `nix develop --command ...`.
- Work in the current checkout; do not create a worktree or branch.
- Keep every file below 300 lines and the folder below 16 files.
- `max_silence_gap` defaults to `80` ms and accepts integers greater than or equal to zero.
- `word_overlap_drop_ratio` defaults to `0.5` and accepts `0.0..1.0`.
- Drop only when union word overlap divided by clipped silence length is strictly greater than the configured ratio.
- Preserve duplicate breaks across adjacent segments.
- Do not retain temporary tests, workflows, audio, or run outputs.

---

### Task 1: Bridge short gaps between silence intervals

**Files:**
- Modify: `src/runner/nodes/audio_segments/silence_detection.py`
- Test temporarily: `/tmp/test_silence_gap_overlap.py`

**Interfaces:**
- Changes: `detect_silence_intervals(data: bytes, audio_start: float, threshold: float, window_size_ms: int, max_gap_ms: int) -> list[SilenceInterval]`.
- Changes: `_merged_intervals(..., max_gap_samples: int) -> list[SilenceInterval]`.

- [ ] **Step 1: Write the failing gap regression**

Create an in-memory 1 kHz WAV containing 280 ms silence, 60 ms tone, and 160 ms silence. Call:

```python
merged = detect_silence_intervals(wav_bytes, 0.0, 0.01, 20, 80)
assert merged == [SilenceInterval(0.0, 0.5)]
assert len(detect_silence_intervals(wav_bytes, 0.0, 0.01, 20, 59)) == 2
assert len(detect_silence_intervals(wav_bytes, 0.0, 0.01, 20, 60)) == 1
assert len(detect_silence_intervals(wav_bytes, 0.0, 0.01, 20, 0)) == 2
```

- [ ] **Step 2: Run RED**

Run `nix develop --command python /tmp/test_silence_gap_overlap.py`; expect `TypeError` because the detector does not accept `max_gap_ms`.

- [ ] **Step 3: Implement gap-aware merging**

Pass `max_gap_ms` into `_merged_intervals` as rounded samples:

```python
max_gap_samples = int(round(int(sample_rate) * max_gap_ms / 1000.0))
return _merged_intervals(silent_windows, len(mono), int(sample_rate), audio_start, max_gap_samples)
```

Bridge intervals when the sample gap is within the setting:

```python
for start, end in windows[1:]:
    if start - current_end <= max_gap_samples:
        current_end = end
    else:
        merged.append((current_start, current_end))
        current_start, current_end = start, end
```

- [ ] **Step 4: Run GREEN**

Run `nix develop --command python /tmp/test_silence_gap_overlap.py`; expect all four gap assertions to pass.

### Task 2: Reject excessive union overlap with words

**Files:**
- Modify: `src/runner/nodes/audio_segments/break_alignment.py`
- Modify: `src/runner/nodes/audio_segments/silence_breaks.py`
- Extend temporarily: `/tmp/test_silence_gap_overlap.py`

**Interfaces:**
- Changes: `annotate_segment(..., drop_prob: float, word_overlap_drop_ratio: float, random_value=...) -> AudioSegment`.
- Produces: `_word_overlap_duration(interval: SilenceInterval, words: list[AlignedWord]) -> float`.
- Adds settings: `max_silence_gap: int = 80` and `word_overlap_drop_ratio: float = 0.5`.

- [ ] **Step 1: Write failing overlap and schema regressions**

Use two aligned words around a silence interval and assert:

```python
kept = annotate_segment(exact_half_segment, [SilenceInterval(1.5, 4.5)], 100, False, False, 0.0, 0.5)
assert "<break t=3000>" in kept.text
dropped = annotate_segment(over_half_segment, [SilenceInterval(1.5, 4.5)], 100, False, False, 0.0, 0.5)
assert dropped == over_half_segment
```

Give `over_half_segment` overlapping word timings whose raw intersection sum exceeds the interval but union overlap is 50%; assert the candidate remains eligible. Validate defaults and bounds:

```python
settings = InsertSilenceBreaksSettings()
assert settings.max_silence_gap == 80
assert settings.word_overlap_drop_ratio == 0.5
for value in (-0.01, 1.01):
    try:
        InsertSilenceBreaksSettings(word_overlap_drop_ratio=value)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid word overlap ratio accepted")
```

- [ ] **Step 2: Run RED**

Run `nix develop --command python /tmp/test_silence_gap_overlap.py`; expect missing settings or an `annotate_segment` signature failure.

- [ ] **Step 3: Implement union overlap filtering**

In `_select_candidate`, after clipping the interval and before minimum-duration and boundary logic, add:

```python
duration = interval.end - interval.start
if _word_overlap_duration(interval, words) / duration > word_overlap_drop_ratio:
    return None
```

Compute the union without double-counting overlapping words:

```python
def _word_overlap_duration(interval: SilenceInterval, words: list[AlignedWord]) -> float:
    intersections = sorted(
        (max(interval.start, word.start), min(interval.end, word.end))
        for word in words
        if _overlap(interval.start, interval.end, word.start, word.end) > 0.0
    )
    total = 0.0
    current_start = current_end = 0.0
    for start, end in intersections:
        if total == 0.0 and current_end == 0.0:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + max(0.0, current_end - current_start)
```

Add Pydantic fields:

```python
max_silence_gap: int = Field(default=80, ge=0, title="Maximum silence gap (ms)")
word_overlap_drop_ratio: float = Field(default=0.5, ge=0.0, le=1.0, title="Word overlap drop ratio")
```

Pass both settings through the detector and annotator call sites.

- [ ] **Step 4: Run GREEN and source checks**

Run the temporary script, compile both audio-segment modules, check both modified files are below 300 lines, and run `git diff --check`. Expect every command to exit zero.

### Task 3: Known-row graph verification and cleanup

**Files:**
- Create temporarily: `/tmp/silence_gap_graph.json`
- Remove: `/tmp/test_silence_gap_overlap.py`
- Remove: `/tmp/silence_gap_graph.json`

**Interfaces:**
- Verifies: `HetznerDsV2Source(row_offset=72, row_limit=1) -> InsertSilenceBreaks` through `POST /graphs/runs`.

- [ ] **Step 1: Verify the known audio through helpers**

Load cached metadata and Parquet row 72 with `audio_from_row`, detect intervals with threshold `0.01`, window `20`, and gap `80`, then annotate its Parakeet segment with ratio `0.5`. Assert no break exists between `Czernichowskiego,` and `a`, and other eligible pauses remain.

- [ ] **Step 2: Submit the registered graph**

Create a non-persisting graph using the interface above and settings `max_silence_gap=80`, `word_overlap_drop_ratio=0.5`, `min_break_time=100`, start/end enabled, and `drop_prob=0.0`. Submit to `POST /graphs/runs`, then inspect its fixed run ID with `nix develop --command python -m cli run silence_gap_word_overlap_verification` and `logs`; expect `succeeded`.

- [ ] **Step 3: Clean and verify**

Remove both temporary files. Compile `src/runner/nodes/audio_segments`, confirm the graph still reports `succeeded`, check file/folder size limits, run `git diff --check`, and inspect `git status --short` without altering unrelated changes.
