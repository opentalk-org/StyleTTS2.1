# DS v2 Alignment Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase Parakeet word timestamps to the selected ds_v2 sample window so padded audio cannot import words outside `text_parakeet`.

**Architecture:** Put source-window extraction and validation in a focused `ds_v2_alignment.py` module, called by `ds_v2_audio.py` where CSV row coordinates and transcript timestamps first become typed audio segments. Represent the source timestamp window explicitly, extract and clip words against that window, then validate the resulting ordered words against the Parakeet transcript before constructing the segment.

**Tech Stack:** Python 3.12, dataclasses, existing runner audio models, Nix development shell, FastAPI graph API, runflow CLI.

## Global Constraints

- Run Python, backend, runner, and CLI commands only through `nix develop --command ...`.
- Work in the current checkout; do not create a worktree or branch.
- Keep every file under 300 lines and every folder under 16 files.
- Keep `Audio.end` and `AudioSegment.end` equal to decoded padded audio duration.
- Bound Parakeet alignment to `sample_start..sample_end`, rebase it to zero, and clip it to `0..(sample_end - sample_start)`.
- Reject missing, reversed, or inconsistent timestamp coordinates with an error identifying the ds_v2 row.
- Compare transcript and alignment after collapsing whitespace; preserve punctuation and letter case.
- Test the registered nodes through a real graph; do not invoke node `execute()` directly.
- Do not retain temporary tests, workflows, audio, or run outputs.

---

### Task 1: Sample-bounded Parakeet alignment

**Files:**
- Modify: `src/runner/nodes/hetzner/ds_v2_audio.py`
- Create: `src/runner/nodes/hetzner/ds_v2_alignment.py`
- Test temporarily: `/tmp/test_ds_v2_alignment_window.py`

**Interfaces:**
- Produces: `AlignmentWindow(source_start: float, source_end: float)` with `duration: float`.
- Produces: `_alignment_window(row: dict[str, Any], row_index: int) -> AlignmentWindow`.
- Changes: `_alignment_from_timestamps(timestamps: Any, window: AlignmentWindow) -> list[dict[str, Any]] | None`.
- Produces: `_validate_alignment_text(text: str, alignment: list[dict[str, Any]] | None, row_index: int) -> None`.

- [ ] **Step 1: Write the failing regression script**

Create `/tmp/test_ds_v2_alignment_window.py` using the known failing boundary:

```python
from math import isclose

from runner.nodes.hetzner.ds_v2_alignment import (
    _alignment_from_timestamps,
    _alignment_window,
    _validate_alignment_text,
)

row = {
    "chunk_start": "27.08",
    "sample_start": "29.80",
    "sample_end": "34.44",
}
timestamps = [
    {"word": "około", "start": 2.72, "end": 3.28},
    {"word": "smoleńskiej,", "start": 6.80, "end": 7.36},
    {"word": "toczonej", "start": 7.44, "end": 8.00},
]
window = _alignment_window(row, 3)
alignment = _alignment_from_timestamps(timestamps, window)
assert isclose(window.duration, 4.64)
assert [entry["word"] for entry in alignment] == ["około", "smoleńskiej,"]
assert isclose(alignment[-1]["end"], 4.64)
_validate_alignment_text("około smoleńskiej,", alignment, 3)

for invalid_row in (
    {"chunk_start": "27.08", "sample_start": "29.80"},
    {"chunk_start": "27.08", "sample_start": "29.80", "sample_end": "29.70"},
    {"chunk_start": "27.08", "sample_start": "27.00", "sample_end": "29.70"},
):
    try:
        _alignment_window(invalid_row, 3)
    except ValueError as error:
        assert "ds_v2 row 3" in str(error)
    else:
        raise AssertionError("invalid alignment window was accepted")

try:
    _validate_alignment_text(
        "około smoleńskiej,",
        [*alignment, {"word": "toczonej", "start": 4.72, "end": 4.94}],
        3,
    )
except ValueError as error:
    assert "ds_v2 row 3" in str(error)
else:
    raise AssertionError("mismatched Parakeet alignment was accepted")
```

- [ ] **Step 2: Run the script and verify RED**

Run:

```bash
nix develop --command python /tmp/test_ds_v2_alignment_window.py
```

Expected: import failure for `_alignment_window` because the sample-window interface does not exist yet.

- [ ] **Step 3: Implement the explicit alignment window**

Create `ds_v2_alignment.py` with the complete focused implementation:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentWindow:
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


def _alignment_window(row: dict[str, Any], row_index: int) -> AlignmentWindow:
    try:
        chunk_start = float(row["chunk_start"])
        sample_start = float(row["sample_start"])
        sample_end = float(row["sample_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ds_v2 row {row_index} has incomplete alignment window") from error
    if not all(math.isfinite(value) for value in (chunk_start, sample_start, sample_end)):
        raise ValueError(f"ds_v2 row {row_index} has non-finite alignment window")
    if sample_start < chunk_start or sample_end < sample_start:
        raise ValueError(
            f"ds_v2 row {row_index} has invalid alignment window: "
            f"chunk_start={chunk_start}, sample_start={sample_start}, sample_end={sample_end}"
        )
    return AlignmentWindow(sample_start - chunk_start, sample_end - chunk_start)


def _alignment_from_timestamps(
    timestamps: Any,
    window: AlignmentWindow,
) -> list[dict[str, Any]] | None:
    if isinstance(timestamps, dict):
        timestamps = timestamps["word"]
    if not isinstance(timestamps, list):
        return None
    alignment = []
    for item in timestamps:
        word = str(item["word"]).strip()
        source_start = float(item["start"])
        source_end = float(item["end"])
        if not word or source_start >= window.source_end or source_end <= window.source_start:
            continue
        start = max(0.0, source_start - window.source_start)
        end = min(window.duration, source_end - window.source_start)
        alignment.append({"word": word, "start": start, "end": max(start, end)})
    return alignment or None


def _validate_alignment_text(
    text: str,
    alignment: list[dict[str, Any]] | None,
    row_index: int,
) -> None:
    transcript = " ".join(text.split())
    aligned = " ".join(str(entry["word"]).strip() for entry in alignment or [])
    if aligned != transcript:
        raise ValueError(
            f"ds_v2 row {row_index} Parakeet alignment does not match transcript: "
            f"transcript={transcript!r}, alignment={aligned!r}"
        )
```

Import the three helpers at the top of `ds_v2_audio.py`:

```python
from runner.nodes.hetzner.ds_v2_alignment import (
    _alignment_from_timestamps,
    _alignment_window,
    _validate_alignment_text,
)
```

Remove its `_window_start` and `_alignment_from_timestamps` functions. Replace the timestamp setup in `_transcript_segment` with:

```python
timestamps = _json_or_text(row["text_timestamps"]) if column == "text_parakeet" else None
alignment = None
if timestamps is not None:
    alignment = _alignment_from_timestamps(timestamps, _alignment_window(row, row_index))
    _validate_alignment_text(text, alignment, row_index)
```

Other transcript sources retain `alignment=None`.

- [ ] **Step 4: Run the regression script and verify GREEN**

Run:

```bash
nix develop --command python /tmp/test_ds_v2_alignment_window.py
```
Expected: exit zero with every boundary and error assertion passing.

- [ ] **Step 5: Check focused source quality**

Run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/hetzner
wc -l src/runner/nodes/hetzner/ds_v2_audio.py
wc -l src/runner/nodes/hetzner/ds_v2_alignment.py
git diff --check -- src/runner/nodes/hetzner/ds_v2_audio.py
git diff --check -- src/runner/nodes/hetzner/ds_v2_alignment.py
```

Expected: compile and diff checks exit zero, and both Python files remain below 300 lines.

### Task 2: Real graph regression and cleanup

**Files:**
- Create temporarily: `/tmp/ds_v2_silence_break_graph.json`
- Remove: `/tmp/test_ds_v2_alignment_window.py`
- Remove: `/tmp/ds_v2_silence_break_graph.json`

**Interfaces:**
- Verifies: `HetznerDsV2Source.audio -> InsertSilenceBreaks.audio` through `POST /graphs/runs`.
- Uses: source `row_offset=3`, `row_limit=1`, `import_audio=true`, and `text_column="text_parakeet"`.

- [ ] **Step 1: Build a non-persisting graph request**

Create this exact non-persisting request:

```json
{
  "run_id": "ds_v2_alignment_window_verification",
  "nodes": [
    {"id": "source", "type": "HetznerDsV2Source", "params": {
      "host": "hetzner-storagebox", "row_offset": 3, "row_limit": 1,
      "import_audio": true, "text_column": "text_parakeet",
      "name_prefix": "ds_v2_alignment_test", "download_retries": 3, "create_voices": false}},
    {"id": "breaks", "type": "InsertSilenceBreaks", "params": {
      "silence_threshold": 0.01, "window_size": 20, "min_break_time": 100,
      "insert_at_start": true, "insert_at_end": true, "drop_prob": 0.0}}
  ],
  "edges": [{"source_node": "source", "source_port": "audio",
    "target_node": "breaks", "target_port": "audio"}],
  "context": {"work_dir": "work", "cache_dir": "cache", "output_dir": "outputs",
    "device": "cuda", "config": {"resources": {"io": 1.0, "cpu_workers": 1.0}}, "input_items": []}
}
```

- [ ] **Step 2: Submit and inspect the graph**

Confirm the shared stack, submit the graph, and inspect its fixed run ID:

```bash
nix develop --command runflow-dev-status
curl -fsS -H 'Content-Type: application/json' --data-binary @/tmp/ds_v2_silence_break_graph.json http://127.0.0.1:8001/graphs/runs
nix develop --command python -m cli run ds_v2_alignment_window_verification
nix develop --command python -m cli logs ds_v2_alignment_window_verification
```

Expected: the run reaches `succeeded`; `InsertSilenceBreaks` no longer raises for `toczonej`.

- [ ] **Step 3: Verify the original row conversion**

Extend the temporary regression script with the same cached metadata row and the public row-to-audio conversion helper:

```python
import csv
from itertools import islice
from pathlib import Path
import pyarrow.parquet as pq
from runner.nodes.hetzner.ds_v2_audio import DsV2AudioOptions, audio_from_row
cache = Path("cache/hetzner")
stem = "ds_v2_a3c13ae714734cc1_000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed"
metadata_path, parquet_path = cache / f"{stem}_metadata.csv", cache / f"{stem}.parquet"
with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
    metadata_row = next(islice(csv.DictReader(metadata_file), 3, 4))
audio_batch = next(pq.ParquetFile(parquet_path).iter_batches(batch_size=4, columns=["audio"]))
row = {**metadata_row, "audio": audio_batch.to_pylist()[3]["audio"]}
remote_path = "/home/ds_v2/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet"
audio = audio_from_row(row, DsV2AudioOptions("hetzner-storagebox", remote_path, "text_parakeet", "test"), 3, None)
segment = next(item for item in audio.segments if item.metadata["type_"] == "parakeet")
assert isclose(segment.end, 4.94) and segment.alignment[-1]["word"] == "smoleńskiej,"
assert isclose(segment.alignment[-1]["end"], 4.64)
assert " ".join(entry["word"] for entry in segment.alignment) == " ".join(segment.text.split())
```

Run `nix develop --command python /tmp/test_ds_v2_alignment_window.py`; expect exit zero.

- [ ] **Step 4: Remove temporary artifacts and perform final verification**

Remove both `/tmp` files. Run `nix develop --command python -m compileall -q src/runner/nodes/hetzner src/runner/nodes/audio_segments`, `nix develop --command runflow-dev-status`, `git diff --check`, and `git status --short`.

Expected: compilation and diff checks exit zero, the shared session is running, no temporary artifacts remain in the repository, and unrelated dirty-worktree files are unchanged.
