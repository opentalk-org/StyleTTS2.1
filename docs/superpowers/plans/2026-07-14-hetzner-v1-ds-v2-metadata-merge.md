# Hetzner v1 ds_v2 Metadata Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich long ds_v1 audio with matched ds_v2 recording metadata, absolute-time transcript segments, and Parakeet word alignment.

**Architecture:** Keep `HetznerDsV1ParquetAudioSource` as the orchestration boundary. Extract its existing SFTP/cache code, load and index the corresponding ds_v2 CSV in a focused metadata module, and construct absolute segments in a separate projection module before emitting each `Audio`.

**Tech Stack:** Python 3.12, Pydantic settings, PyArrow, runflow nodes, existing Hetzner SFTP cache helpers, pytest through Nix.

## Global Constraints

- Run every Python and test command through `nix develop --command ...`.
- Do not retain test files; create temporary regression tests and remove them before each production commit.
- Keep every file below 300 lines and the Hetzner folder below 16 files.
- Do not load ds_v2 audio bytes or add audio-specific behavior to `src/runflow`.
- Preserve unrelated working-tree changes.

---

### Task 1: Extract ds_v1 storage access

**Files:**
- Create: `src/runner/nodes/hetzner/ds_v1_storage.py`
- Modify: `src/runner/nodes/hetzner/ds_v1_parquet.py`
- Temporary test: `tmp_tests/test_ds_v1_storage.py`

**Interfaces:**
- Produces: `parquet_path(settings: HetznerDsV1ParquetAudioSourceSettings, context: Any) -> Path`.
- Preserves: current cache naming, retry count, temporary-file cleanup, and SFTP error text.

- [ ] **Step 1: Write the failing import and cache-path test**

```python
from runner.nodes.hetzner.ds_v1_storage import cache_name


def test_cache_name_is_stable_and_keeps_basename():
    value = cache_name("/home/ds_v1/example.parquet")
    assert value.endswith("_example.parquet")
    assert value == cache_name("/home/ds_v1/example.parquet")
```

- [ ] **Step 2: Verify the test fails before the module exists**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_storage.py`

Expected: collection error containing `No module named 'runner.nodes.hetzner.ds_v1_storage'`.

- [ ] **Step 3: Move storage code without changing behavior**

Move `_parquet_path`, `_cache_name`, `_download_sftp_file`, and
`_sftp_error_detail` to `ds_v1_storage.py`, rename the first two without leading
underscores, and import `parquet_path` in `ds_v1_parquet.py`. Keep the exact SFTP
command, retry loop, and error construction. Change `_load_audio_items` to:

```python
def _load_audio_items(settings, context):
    local_path = parquet_path(settings, context)
    rows = list(_iter_parquet_rows(local_path, settings.row_offset, settings.row_limit))
    return [_audio_from_row(row, settings, absolute_index) for absolute_index, row in rows]
```

- [ ] **Step 4: Verify the extraction and file limits**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_storage.py`

Expected: `1 passed`.

Run: `wc -l src/runner/nodes/hetzner/ds_v1_parquet.py src/runner/nodes/hetzner/ds_v1_storage.py`

Expected: both counts below 300.

- [ ] **Step 5: Remove the temporary test and commit scoped files**

```bash
# Delete tmp_tests/test_ds_v1_storage.py with apply_patch.
git add src/runner/nodes/hetzner/ds_v1_parquet.py src/runner/nodes/hetzner/ds_v1_storage.py
git commit -m "refactor: isolate hetzner v1 storage access"
```

### Task 2: Load, match, and merge ds_v2 metadata

**Files:**
- Create: `src/runner/nodes/hetzner/ds_v1_metadata.py`
- Temporary test: `tmp_tests/test_ds_v1_metadata.py`

**Interfaces:**
- Produces: `DsV2Sample(row_index: int, values: dict[str, str])`.
- Produces: `DsV2MetadataIndex(remote_path: str, samples_by_recording: dict[str, tuple[DsV2Sample, ...]])`.
- Produces: `metadata_path_for_v1(remote_parquet_path: str) -> str`.
- Produces: `read_metadata_index(path: Path, remote_path: str, check_cancel: Callable[[], None]) -> DsV2MetadataIndex`.
- Produces: `load_metadata_index(host: str, remote_v1_path: str, cache_dir: Path, retries: int, check_cancel: Callable[[], None]) -> DsV2MetadataIndex`.
- Produces: `matching_samples(index: DsV2MetadataIndex, row: dict[str, Any], row_index: int) -> tuple[DsV2Sample, ...]`.
- Produces: `merge_recording_metadata(metadata: dict[str, Any], samples: tuple[DsV2Sample, ...], remote_path: str) -> dict[str, Any]`.

- [ ] **Step 1: Write failing focused metadata tests**

Create a CSV fixture with `CSV_METADATA_COLUMNS` and two rows for
`video.opus`. Assert:

```python
assert metadata_path_for_v1("/home/ds_v1/batch.parquet") == \
    "/home/ds_v2_metadata/batch_processed_metadata.csv"
assert [sample.row_index for sample in matching_samples(index, {"opus_file": "video.opus", "video_id": "video"}, 4)] == [0, 1]
assert merge_recording_metadata({"video_id": "video", "title": None}, samples, index.remote_path)["title"] == "Title from v2"
assert merge_recording_metadata({"video_id": "video", "title": "v1 title"}, samples, index.remote_path)["title"] == "v1 title"
```

Also assert a mismatched embedded `metadata.video_id` raises `ValueError` with
both row indices and `video_id`, and an unmatched v1 row returns an empty tuple.

- [ ] **Step 2: Verify failure is caused by missing metadata module**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_metadata.py`

Expected: collection error for `runner.nodes.hetzner.ds_v1_metadata`.

- [ ] **Step 3: Implement exact discovery and indexing**

Use `PurePosixPath(remote_v1_path).stem` to form the approved metadata path.
Use `cached_remote_file` from `ds_v2_rows.py`, then `csv.DictReader`,
`validate_metadata_headers`, and `parse_metadata_row`. Normalize identifiers
with `PurePosixPath(value).name`. Require ds_v2 `audio_path` and `filename` to
normalize to the same value. Store each CSV row once, ordered by row index, and
call `check_cancel()` during iteration.

- [ ] **Step 4: Implement key-aware recording merge**

Decode each sample's `metadata` JSON object. Require it to be a dictionary and
require all matched rows to agree. Validate `video_id` against the ds_v1 value.
For every embedded key, fill the ds_v1 key only when absent or `None`; retain
non-null ds_v1 descriptive values. Add exactly:

```python
merged["ds_v2_metadata_path"] = remote_path
merged["ds_v2_sample_count"] = len(samples)
```

For no samples, add only `ds_v2_sample_count = 0`.

- [ ] **Step 5: Verify metadata behavior and limits**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_metadata.py`

Expected: all tests pass.

Run: `wc -l src/runner/nodes/hetzner/ds_v1_metadata.py`

Expected: below 300.

- [ ] **Step 6: Remove the temporary test and commit**

```bash
# Delete tmp_tests/test_ds_v1_metadata.py with apply_patch.
git add src/runner/nodes/hetzner/ds_v1_metadata.py
git commit -m "feat: index ds v2 metadata for hetzner v1"
```

### Task 3: Project ds_v2 transcripts onto ds_v1 audio

**Files:**
- Create: `src/runner/nodes/hetzner/ds_v1_segments.py`
- Modify: `src/runner/nodes/hetzner/ds_v1_parquet.py`
- Temporary test: `tmp_tests/test_ds_v1_segments.py`

**Interfaces:**
- Consumes: `DsV2Sample`, `load_metadata_index`, `matching_samples`, and `merge_recording_metadata` from Task 2.
- Produces: `segments_from_samples(audio: Audio, samples: tuple[DsV2Sample, ...], remote_v1_path: str, row_index: int, preferred_text_column: str) -> list[AudioSegment]`.

- [ ] **Step 1: Write failing absolute-projection tests**

Build a 60-second `Audio` and one sample with `sample_start=10`,
`sample_end=14`, `duration=4.3`, all four transcripts, and Parakeet timestamps
whose exact transcript match is local to the sample. Assert four segments in
`src`, `whisper`, `parakeet`, `canary` order; bounds `10.0..14.3`; the Parakeet
word alignment starts at `10 + local_start`; sample fields and decoded source
metadata are present; and every segment uses the v1 `audio_file_id`, rate, and
channels. Add cases for no exact Parakeet match (`alignment is None`) and
`sample_end > audio.duration` (`ValueError`).

- [ ] **Step 2: Verify failure is caused by the missing projection module**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_segments.py`

Expected: collection error for `runner.nodes.hetzner.ds_v1_segments`.

- [ ] **Step 3: Implement typed sample conversion**

Reuse `TRANSCRIPT_SEGMENTS`, `alignment_window`, and
`alignment_from_timestamps`. Parse finite numeric timing fields and require
`0 <= sample_start <= sample_end <= audio.duration` and `duration > 0`.
Construct segment end as `min(audio.duration, sample_start + duration)`. For
Parakeet, add `sample_start` to every local alignment entry and assert ordered
word times inside `sample_start..sample_end`. Stable IDs include the v1 path,
v1 row, ds_v2 row, and transcript source.

- [ ] **Step 4: Integrate metadata loading once per node lifecycle**

Add `text_column` with the same four-value `Literal` and default `text_src` as
the ds_v2 source. During the first execute, load the ds_v2 index once. For every
v1 row, decode the audio, find samples, replace its metadata with the key-aware
merge, and attach projected segments. Call `context.check_cancel()` between
metadata loading, selected rows, and audio decoding.

- [ ] **Step 5: Verify focused tests and schema discovery**

Run: `nix develop --command pytest -q tmp_tests/test_ds_v1_segments.py`

Expected: all tests pass.

Run: `curl -fsS http://127.0.0.1:8001/schema >/tmp/runflow-schema.json`

Expected: exit 0 and `HetznerDsV1ParquetAudioSource` exposes `text_column`.

- [ ] **Step 6: Remove the temporary test and commit**

```bash
# Delete tmp_tests/test_ds_v1_segments.py with apply_patch.
git add src/runner/nodes/hetzner/ds_v1_parquet.py src/runner/nodes/hetzner/ds_v1_segments.py
git commit -m "feat: apply ds v2 transcripts to hetzner v1 audio"
```

### Task 4: Persist and verify the enriched graph

**Files:**
- Modify: `workflows/ds_v1_sample_import.json`
- Modify: `workflows/README.md`

**Interfaces:**
- Consumes: enriched `Audio.segments` emitted by Task 3.
- Produces: a smoke graph that saves the v1 audio record and replaces its stored segments.

- [ ] **Step 1: Update the workflow graph**

Add `text_column: "text_src"` to the v1 source. Add a `SaveAudioSegments` node
with `mode: "replace"`, connect `SaveAudioRecord.audio` to it, and set the
record storage mode explicitly to `stored`. Update the README to state that the
v1 workflow enriches long audio with matched ds_v2 transcripts and absolute
Parakeet alignment.

- [ ] **Step 2: Validate repository files**

Run: `nix develop --command python -m json.tool workflows/ds_v1_sample_import.json >/dev/null`

Expected: exit 0.

Run: `git diff --check && find src/runner/nodes/hetzner -maxdepth 1 -type f -name '*.py' | wc -l`

Expected: no diff errors and no more than 16 Python files.

- [ ] **Step 3: Run the real smoke graph**

Run `nix develop --command runflow-dev-status`; attach/start only through
`nix develop --command runflow-dev-session` if needed. Submit
the workflow's `.data` with `run_id` set to
`hetzner_v1_ds_v2_merge_verification` through `POST /graphs/runs`, then run:

```bash
nix develop --command python -m cli runs
nix develop --command python -m cli logs hetzner_v1_ds_v2_merge_verification
```

Expected: the run completes; the source reports a positive
`ds_v2_sample_count`; saved segments contain all available transcript variants;
and Parakeet word times are absolute and within the long recording.

- [ ] **Step 4: Commit workflow documentation**

```bash
git add workflows/ds_v1_sample_import.json workflows/README.md
git commit -m "docs: verify enriched hetzner v1 import workflow"
```

- [ ] **Step 5: Run final scoped verification**

Run schema export, JSON validation, `git diff --check`, and inspect
`git status --short`. Confirm no temporary test created by this plan remains and
all unrelated files are untouched.
