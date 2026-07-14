# Deduplicate Transcript-Aligned Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every non-null alignment emitted by `DeduplicateOverlappingSegments` match the selected consensus transcription token-for-token.

**Architecture:** Add a focused transcript-alignment projector beside the existing alignment merge helpers; it maps every member track onto the winner's token skeleton with order-preserving dynamic programming, selects the best timing per position, and interpolates missing positions. Replace only overlap-deduplication's union merge; keep the shared `MergeAlignment` API and behavior unchanged.

**Tech Stack:** Python 3.12, dataclasses, Pydantic/runflow node runtime, temporary regression scripts, Nix development shell and runflow CLI.

## Global Constraints

- Run Python, backend, runner, and CLI commands through `nix develop --command ...`.
- Work in the current checkout; do not create a worktree or branch.
- Keep every file below 300 lines and every folder below 16 files.
- Output alignment words must equal `winner.text.strip().split()` exactly.
- Match lowercase alphanumeric-normalized words in order and one-to-one.
- Prefer numeric score, then winner track, then member order.
- Interpolate every unmatched transcript token and mark it with `interpolated: true`.
- Empty winner text produces `alignment=None`.
- Do not change `MergeAlignment` behavior or retain temporary files.

---

### Task 1: Transcript-skeleton projection

**Files:**
- Modify: `src/runner/nodes/audio_segments/alignment_merge.py`
- Test temporarily: `/tmp/test_dedup_transcript_alignment.py`

**Interfaces:**
- Produces: `AlignmentTrack(words: list[dict[str, Any]], preferred: bool = False)`.
- Produces: `project_alignment_to_transcript(text: str, segment_start: float, segment_end: float, tracks: list[AlignmentTrack]) -> tuple[list[dict[str, Any]] | None, int]`.

- [ ] **Step 1: Write the failing projector regression**

Use skeleton `"Hello, go go now."` with a winner track missing the second `go` and another track containing `hello`, both `go` occurrences, `extra`, and `now`. Assert:

```python
alignment, interpolated = project_alignment_to_transcript(text, 0.0, 4.0, tracks)
assert [entry["word"] for entry in alignment] == ["Hello,", "go", "go", "now."]
assert "extra" not in [entry["word"] for entry in alignment]
assert interpolated == 0
```

Add cases proving a higher numeric score wins, a score tie prefers the winner track, punctuation/case comes from the transcript, all-unmatched text partitions the full segment, and empty text returns `(None, 0)`.

- [ ] **Step 2: Run RED**

Run `nix develop --command python /tmp/test_dedup_transcript_alignment.py`; expect module import failure.

- [ ] **Step 3: Implement order-preserving matching**

Define frozen `AlignmentTrack` and an internal candidate carrying entry, preferred flag, and track index. Build an LCS table per track using:

```python
if normalized_token == normalized_word:
    table[token_index][word_index] = 1 + table[token_index + 1][word_index + 1]
else:
    table[token_index][word_index] = max(
        table[token_index + 1][word_index],
        table[token_index][word_index + 1],
    )
```

Walk the table deterministically to return one-to-one `(token_index, word_index)` mappings. Collect candidates per token and rank with:

```python
score = float(entry["score"]) if entry.get("score") is not None else float("-inf")
rank = (-score, not preferred, track_index)
```

Clamp finite candidate timings to segment bounds, require `start <= end`, and process skeleton positions in order so a candidate with a start before the previous selected start is left unmatched for interpolation. Copy auxiliary fields but replace `word` with the exact skeleton token.

- [ ] **Step 4: Implement interpolation and invariants**

For each contiguous unmatched run, evenly partition the positive span bounded by the previous entry's end/segment start and next entry's start/segment end. For a non-positive internal span, place zero-duration entries at evenly spaced points between the neighboring midpoints. Mark each generated entry `interpolated: true`.

Validate exact words, finite `start/end`, segment bounds, `start <= end`, and nondecreasing starts before returning `(alignment, interpolation_count)`.

- [ ] **Step 5: Run GREEN**

Run the temporary script; expect every mapping, selection, interpolation, and empty-text assertion to pass.

### Task 2: Integrate projection into overlap consensus

**Files:**
- Modify: `src/runner/nodes/statistics/segments.py`
- Modify: `src/runner/nodes/audio_segments/dedup_overlap.py`
- Extend temporarily: `/tmp/test_dedup_transcript_alignment.py`

**Interfaces:**
- Replaces overlap consensus use of `merge_alignment_tracks` with `project_alignment_to_transcript`.
- Adds metadata key `alignment_interpolated_words: int`.

- [ ] **Step 1: Write failing consensus regressions**

Construct overlapping `AudioSegment` members where the winning text omits a losing-track extra, contains a repeated word absent from the winner alignment, and has one token absent from every track. Call `deduplicate_overlapping_segments` and assert the kept segment's alignment words exactly equal `kept.text.split()`, the extra is absent, the missing token is interpolated, and metadata reports one interpolated word.

- [ ] **Step 2: Run RED**

Run the temporary script; expect the current merged alignment to contain the losing-track extra or the existing merge call to fail.

- [ ] **Step 3: Wire the projector into `_consensus_segment`**

Import `AlignmentTrack` and `project_alignment_to_transcript`. Build the first preferred track from `winner.alignment`, followed by other member words whose midpoint lies inside the winner span. Project against `winner.text`, store `alignment_interpolated_words` in consensus metadata, and return the projected alignment.

Update the node description to state that output alignments are transcript-constrained and missing timings are interpolated.

- [ ] **Step 4: Run GREEN and static checks**

Run the temporary regression, compile the affected modules, confirm all files and the folder remain within limits, and run `git diff --check`; expect success.

### Task 3: Registered graph verification and cleanup

**Files:**
- Create temporarily: `/tmp/dedup_transcript_alignment_graph.json`
- Remove: `/tmp/test_dedup_transcript_alignment.py`
- Remove: `/tmp/dedup_transcript_alignment_graph.json`

**Interfaces:**
- Verifies: `AudioSource -> LoadAudioSegments -> DeduplicateOverlappingSegments` through `POST /graphs/runs`.

- [ ] **Step 1: Select an overlapping stored record and submit the graph**

Use read-only API inspection to select one record containing overlapping transcript alternatives. Create a non-persisting graph limited to that record, submit it through `POST /graphs/runs`, and inspect its fixed run ID with the CLI.

- [ ] **Step 2: Verify output invariant and clean up**

Exercise the same stored members through the temporary helper regression and assert every non-null kept alignment matches `text.strip().split()`. Remove both temporary files, rerun compilation and CLI status, check size limits and `git diff --check`, and preserve unrelated dirty changes.
