# Silence Break Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batch-aware `InsertSilenceBreaks` node that annotates aligned audio segments from fixed-window RMS silence detection, and convert `PadSilence` to a dBFS threshold.

**Architecture:** Keep waveform analysis, alignment/text annotation, and runtime node wiring in three focused modules under `audio_segments`. Decode each audio once, detect maximal silent intervals on the audio timeline, then map each interval to at most one eligible boundary per aligned segment. `PadSilence` receives a localized schema and threshold conversion change.

**Tech Stack:** Python 3.11+, Pydantic settings, NumPy, SoundFile, runflow nodes/ports/policies, pytest for temporary helper verification, Nix-managed runtime and CLI for graph verification.

## Global Constraints

- Keep every file below 300 lines and `src/runner/nodes/audio_segments/` at no more than 16 files.
- Run Python, pytest, backend, runner, and CLI commands only through `nix develop --command ...`.
- Do not create a worktree or branch; preserve all unrelated dirty-worktree changes.
- Do not commit temporary tests, WAVs, workflows, or run outputs; remove them before completion.
- Test the node through a registered graph, never by directly invoking `execute()`.
- `silence_threshold` is linear RMS in `0.0..1.0`; `PadSilence.silence_threshold_db` is dBFS in `-80.0..0.0`.
- A silence produces at most one break per segment, while adjacent segments may each receive their own clipped boundary break.

---

### Task 1: Fixed-window silence detection
**Files:**
- Create: `src/runner/nodes/audio_segments/silence_detection.py`
- Test temporarily: `.tmp_tests/test_silence_detection.py`

**Interfaces:**
- Produces: `SilenceInterval(start: float, end: float)`.
- Produces: `detect_silence_intervals(data: bytes, audio_start: float, threshold: float, window_size_ms: int, dependencies: tuple[Any, Any] | None = None) -> list[SilenceInterval]`.
- [ ] **Step 1: Write failing helper tests**

Create temporary tests using an in-memory WAV with 100 ms tone, 250 ms zeroes,
and 100 ms tone. Assert that 50 ms non-overlapping windows merge the middle
silence, that timestamps include `audio_start`, and that a final partial silent
window ends at the exact audio duration.

```python
intervals = detect_silence_intervals(wav_bytes, 2.0, 0.01, 50)
assert intervals == [SilenceInterval(start=2.1, end=2.35)]
```

- [ ] **Step 2: Run the tests and observe the missing module failure**

Run: `nix develop --command pytest -q .tmp_tests/test_silence_detection.py`

Expected: collection fails because `silence_detection` does not exist.

- [ ] **Step 3: Implement detection**

Use frozen `SilenceInterval`, SoundFile decoding with `always_2d=True`, channel
mean to mono, and NumPy float64 power. Compute `window_samples` with rounded
milliseconds, include the final partial window, merge consecutive RMS values
at or below the threshold, and clamp timestamps to decoded duration.

```python
@dataclass(frozen=True)
class SilenceInterval:
    start: float
    end: float

def detect_silence_intervals(data: bytes, audio_start: float, threshold: float,
                             window_size_ms: int, dependencies=None) -> list[SilenceInterval]:
    np, sf = dependencies if dependencies is not None else _audio_dependencies()
    samples, sample_rate = sf.read(BytesIO(data), always_2d=True, dtype="float32")
    mono = np.mean(samples, axis=1, dtype=np.float64)
    window_samples = max(1, int(round(int(sample_rate) * window_size_ms / 1000.0)))
    intervals: list[SilenceInterval] = []
    silent_start: int | None = None
    for start in range(0, len(mono), window_samples):
        end = min(len(mono), start + window_samples)
        rms = float(np.sqrt(np.mean(np.square(mono[start:end]))))
        if rms <= threshold and silent_start is None:
            silent_start = start
        if rms > threshold and silent_start is not None:
            intervals.append(SilenceInterval(audio_start + silent_start / sample_rate,
                                             audio_start + start / sample_rate))
            silent_start = None
    if silent_start is not None:
        intervals.append(SilenceInterval(audio_start + silent_start / sample_rate,
                                         audio_start + len(mono) / sample_rate))
    return intervals
```

- [ ] **Step 4: Run detection tests**

Run: `nix develop --command pytest -q .tmp_tests/test_silence_detection.py`

Expected: all tests pass.

### Task 2: Segment alignment and transcript annotation
**Files:**
- Create: `src/runner/nodes/audio_segments/break_alignment.py`
- Test temporarily: `.tmp_tests/test_break_alignment.py`

**Interfaces:**
- Consumes: `SilenceInterval` from Task 1.
- Produces: `annotate_segment(segment: AudioSegment, silences: list[SilenceInterval], min_break_time_ms: int, insert_at_start: bool, insert_at_end: bool, drop_prob: float, random_value: Callable[[], float] = random.random) -> AudioSegment`.
- [ ] **Step 1: Write failing mapping tests**

Cover an internal gap, greatest-overlap selection across several word gaps,
post-clipping minimum duration, enabled start/end candidates, two adjacent
segments independently receiving boundary breaks, `drop_prob` values `0.0`
and `1.0`, unchanged `None`/empty alignments, transcript token insertion,
chronological alignment insertion, repeat execution, and segment-identified
errors for invalid timing or transcript mismatch.

```python
updated = annotate_segment(segment, [SilenceInterval(1.18, 1.42)], 100, False, False, 0.0)
assert updated.text == "hello <break t=240> world"
assert updated.alignment[1] == {
    "word": "<break t=240>",
    "start": 1.18,
    "end": 1.42,
}
```

- [ ] **Step 2: Run the tests and observe the missing module failure**

Run: `nix develop --command pytest -q .tmp_tests/test_break_alignment.py`

Expected: collection fails because `break_alignment` does not exist.

- [ ] **Step 3: Implement typed boundary selection and annotation**

Define a frozen internal `BreakCandidate` containing boundary index, interval,
and overlap. Exclude existing `<break t=N>` entries from ordinary aligned words,
validate chronological finite timings inside the segment, select one candidate
per silence by `(-overlap, boundary_index)`, apply the random decision, and
deduplicate identical existing breaks. Match aligned word strings sequentially
in `segment.text`, then splice tokens from right to left so repeated words remain
unambiguous and all non-boundary text is preserved.

```python
def annotate_segment(segment: AudioSegment, silences: list[SilenceInterval],
                     min_break_time_ms: int, insert_at_start: bool,
                     insert_at_end: bool, drop_prob: float,
                     random_value: Callable[[], float] = random.random) -> AudioSegment:
    if not segment.alignment:
        return segment
    words = _aligned_words(segment)
    selected = [_select_candidate(segment, words, silence, min_break_time_ms,
                                  insert_at_start, insert_at_end)
                for silence in silences]
    selected = [candidate for candidate in selected if candidate is not None]
    kept = [candidate for candidate in selected if random_value() >= drop_prob]
    if not kept:
        return segment
    return replace(segment, text=_insert_break_text(segment, words, kept),
                   alignment=_insert_break_alignments(segment.alignment, kept))
```

- [ ] **Step 4: Run mapping tests**

Run: `nix develop --command pytest -q .tmp_tests/test_break_alignment.py`

Expected: all tests pass.

### Task 3: Runtime node and registry wiring
**Files:**
- Create: `src/runner/nodes/audio_segments/silence_breaks.py`
- Modify: `src/runner/nodes/registry.py`
- Test temporarily: `.tmp_tests/test_silence_break_node_schema.py`

**Interfaces:**
- Consumes: `detect_silence_intervals` and `annotate_segment`.
- Produces: registered node type `InsertSilenceBreaks` with one `AudioPort` input and output.
- [ ] **Step 1: Write a failing registry/schema test**

Assert that `create_node_registry()` resolves `InsertSilenceBreaks`, rejects
thresholds outside `0.0..1.0`, rejects non-positive millisecond settings, and
rejects `drop_prob` outside `0.0..1.0`.

- [ ] **Step 2: Run the schema test and observe failure**

Run: `nix develop --command pytest -q .tmp_tests/test_silence_break_node_schema.py`

Expected: failure because the node is not registered.

- [ ] **Step 3: Implement the batch-aware node**

Create `InsertSilenceBreaksSettings(StrictSettings)` with exact approved fields.
Use `BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)`. For
each audio, require `Audio` and non-`None` bytes, detect silence once, check
cancellation between audios and segments, annotate every segment, and replace
the audio's segment list without changing waveform bytes.

```python
class InsertSilenceBreaksNode(Node):
    NODE_TYPE = "InsertSilenceBreaks"
    CATEGORY = "Audio"
    SETTINGS = InsertSilenceBreaksSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
```

Import and register the class in `src/runner/nodes/registry.py` without
reformatting or staging unrelated existing changes.

- [ ] **Step 4: Run schema and helper tests together**

Run: `nix develop --command pytest -q .tmp_tests/test_silence_detection.py .tmp_tests/test_break_alignment.py .tmp_tests/test_silence_break_node_schema.py`

Expected: all tests pass.

### Task 4: PadSilence dBFS conversion
**Files:**
- Modify: `src/runner/nodes/audio_enhancement/pad_silence.py`
- Test temporarily: `.tmp_tests/test_pad_silence_dbfs.py`

**Interfaces:**
- Changes: `PadSilenceSettings.silence_threshold_db: float` bounded by `-80.0..0.0`.
- Changes: `_non_silent_content(..., threshold_db: float)` converts with `10.0 ** (threshold_db / 20.0)`.
- [ ] **Step 1: Write failing dBFS tests**

Assert schema acceptance of `-40.0`, rejection of positive values, rejection of
the removed `silence_threshold` field, and trimming equivalence between `-40`
dBFS and linear RMS `0.01` on controlled samples.

- [ ] **Step 2: Run tests and observe schema failure**

Run: `nix develop --command pytest -q .tmp_tests/test_pad_silence_dbfs.py`

Expected: failure because `silence_threshold_db` is not yet defined.

- [ ] **Step 3: Rename and convert the setting**

```python
class PadSilenceSettings(StrictSettings):
    silence_threshold_db: float = Field(ge=-80.0, le=0.0)
    start_silence: int = Field(ge=0, title="Start silence (ms)")
    end_silence: int = Field(ge=0, title="End silence (ms)")


def _non_silent_content(np: Any, samples: Any, sample_rate: int, threshold_db: float) -> Any:
    threshold = 10.0 ** (threshold_db / 20.0)
```

Update call sites and emitted settings metadata. Do not add an alias or fallback.

- [ ] **Step 4: Run all temporary tests**

Run: `nix develop --command pytest -q .tmp_tests`

Expected: all tests pass.

### Task 5: Registered graph verification and cleanup
**Files:**
- Create temporarily: `.tmp_silence_break_graph.json`
- Remove: `.tmp_tests/`, `.tmp_silence_break_graph.json`, and generated WAV/run files.

**Interfaces:**
- Verifies: `AudioSource -> LoadAudio -> LoadAudioSegments -> InsertSilenceBreaks` through `POST /graphs/runs`.
- [ ] **Step 1: Confirm the shared stack and select a stored aligned audio record**

Run: `nix develop --command runflow-dev-status`

Query the existing backend/database through project APIs or CLI for one
non-virtual audio record with bytes and a non-empty alignment. Do not mutate its
stored segments; omit `SaveAudioSegments` from the smoke graph.

- [ ] **Step 2: Submit the temporary graph**

Use `AudioSource(source="selected")`, `LoadAudio`, `LoadAudioSegments`, and
`InsertSilenceBreaks` with a conservative threshold, `window_size=20`,
`min_break_time=100`, both boundary flags enabled, and `drop_prob=0.0`. Submit
the JSON to `POST /graphs/runs` using the same backend endpoint as the UI.

- [ ] **Step 3: Inspect the real run**

Run through Nix:

```bash
python -m cli run <run_id>
python -m cli logs <run_id>
python -m cli failed <run_id>
```

Expected: the run completes, the registered node executes without failure, and
logs contain no traceback.

- [ ] **Step 4: Run final static and focused verification**

Run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/audio_segments src/runner/nodes/audio_enhancement/pad_silence.py
nix develop --command pytest -q .tmp_tests
git diff --check
```

Expected: every command exits zero.
- [ ] **Step 5: Remove temporary verification artifacts and inspect scope**

Delete only the temporary files created by this plan. Confirm the permanent
changes are limited to the three focused audio-segment modules, `registry.py`,
and `pad_silence.py`, plus the approved design and plan documents. Preserve all
pre-existing dirty changes.
