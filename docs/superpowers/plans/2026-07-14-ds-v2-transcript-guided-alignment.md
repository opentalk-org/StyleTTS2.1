# DS v2 Transcript-Guided Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the exact Parakeet timestamp sequence nearest each ds_v2 sample window and emit no alignment when no exact sequence exists.

**Architecture:** Move transcript matching into `alignment_from_timestamps`, replacing overlap-only extraction and post-extraction validation. Score exact contiguous timestamp sequences by their combined start/end distance from the sample window, then rebase and clamp the winning sequence.

**Tech Stack:** Python 3.12, existing runner node graph, Nix development shell, FastAPI graph API.

## Global Constraints

- Run Python and CLI commands through `nix develop --command ...`.
- Keep runtime files below 300 lines and do not retain temporary tests or graph requests.
- Preserve exact case and punctuation when matching timestamp words.
- Invalid sample-window coordinates remain errors.
- Verify registered nodes through real graphs.

---

### Task 1: Exact nearest transcript alignment

**Files:**
- Modify: `src/runner/nodes/hetzner/ds_v2_alignment.py`
- Modify: `src/runner/nodes/hetzner/ds_v2_audio.py`
- Test temporarily: `/tmp/test_ds_v2_transcript_alignment.py`
- Test temporarily: `/tmp/ds_v2_transcript_alignment_graph.json`

**Interfaces:**
- Changes: `alignment_from_timestamps(timestamps: Any, text: str, window: AlignmentWindow) -> list[dict[str, Any]] | None`
- Removes: `validate_alignment_text(...)`

- [ ] **Step 1: Write the failing regression script**

Create a temporary script covering row 138 exclusion, row 42 inclusion, duplicate selection by boundary distance, and no-match `None`:

```python
from math import isclose

from runner.nodes.hetzner.ds_v2_alignment import AlignmentWindow, alignment_from_timestamps

row_138 = [
    {"word": "a", "start": 78.4, "end": 78.64},
    {"word": "katolików", "start": 78.64, "end": 79.52},
]
assert [item["word"] for item in alignment_from_timestamps(
    row_138, "katolików", AlignmentWindow(343.672 - 265.032, 79.52)
) or []] == ["katolików"]

row_42 = [
    {"word": "Na", "start": 76.8, "end": 76.96},
    {"word": "początku", "start": 76.96, "end": 77.52},
]
assert [item["word"] for item in alignment_from_timestamps(
    row_42, "Na początku", AlignmentWindow(371.336 - 294.376, 77.52)
) or []] == ["Na", "początku"]

duplicates = [
    {"word": "tak", "start": 1.0, "end": 1.2},
    {"word": "tak", "start": 8.0, "end": 8.2},
]
chosen = alignment_from_timestamps(duplicates, "tak", AlignmentWindow(7.9, 8.3))
assert chosen is not None and chosen[0]["word"] == "tak"
assert isclose(chosen[0]["start"], 0.1) and isclose(chosen[0]["end"], 0.3)
assert alignment_from_timestamps(duplicates, "nie", AlignmentWindow(7.9, 8.3)) is None
```

- [ ] **Step 2: Verify RED**

Run `nix develop --command python /tmp/test_ds_v2_transcript_alignment.py`.
Expected: `TypeError` because the current function does not accept the transcript.

- [ ] **Step 3: Implement the minimal selection algorithm**

In `ds_v2_alignment.py`, replace overlap extraction with exact contiguous matching:

```python
def alignment_from_timestamps(
    timestamps: Any,
    text: str,
    window: AlignmentWindow,
) -> list[dict[str, Any]] | None:
    if isinstance(timestamps, dict):
        timestamps = timestamps["word"]
    if not isinstance(timestamps, list):
        return None
    transcript_words = text.split()
    if not transcript_words:
        return None
    timestamp_words = [str(item["word"]).strip() for item in timestamps]
    width = len(transcript_words)
    candidates = [
        start
        for start in range(len(timestamps) - width + 1)
        if timestamp_words[start:start + width] == transcript_words
    ]
    if not candidates:
        return None
    selected_start = min(
        candidates,
        key=lambda start: (
            abs(float(timestamps[start]["start"]) - window.source_start)
            + abs(float(timestamps[start + width - 1]["end"]) - window.source_end),
            start,
        ),
    )
    alignment = []
    for item in timestamps[selected_start:selected_start + width]:
        start = min(window.duration, max(0.0, float(item["start"]) - window.source_start))
        end = min(window.duration, max(0.0, float(item["end"]) - window.source_start))
        alignment.append({"word": str(item["word"]).strip(), "start": start, "end": max(start, end)})
    return alignment
```

Delete `validate_alignment_text`. Update `ds_v2_audio.py` to pass `text` into `alignment_from_timestamps` and remove validation.

- [ ] **Step 4: Verify GREEN and source quality**

Run:

```bash
nix develop --command python /tmp/test_ds_v2_transcript_alignment.py
nix develop --command python -m compileall -q src/runner/nodes/hetzner
git diff --check -- src/runner/nodes/hetzner/ds_v2_alignment.py src/runner/nodes/hetzner/ds_v2_audio.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Verify the registered graph and clean up**

Submit source graphs for global offsets 42 and 138 with `row_limit=1`, `import_audio=true`, and `text_column="text_parakeet"`, each connected to `InsertSilenceBreaks`. Confirm both runs succeed with `nix develop --command python -m cli run <run_id>`.

Delete both temporary files, then rerun compilation, `git diff --check`, and `git status --short`.
