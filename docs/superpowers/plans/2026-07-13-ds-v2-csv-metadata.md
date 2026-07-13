# ds_v2 CSV Metadata Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `HetznerDsV2ParquetAudioSource` always pair a cached Hetzner Parquet audio source with its exact `/home/ds_v2_metadata/*_metadata.csv` metadata source and fail on any invalid pair.

**Architecture:** A new row-source module resolves and caches the exact remote pair, streams CSV metadata against lightweight Parquet identity columns for full-file validation, and returns selected rows containing Parquet audio plus CSV metadata. The node module keeps registration, settings, voice creation, and `Audio`/`AudioSegment` conversion while dropping all local and optional-cache paths.

**Tech Stack:** Python 3.12, Pydantic, PyArrow Parquet, standard-library `csv`, SFTP subprocess, runflow node runtime.

## Global Constraints

- Run Python, CLI, and graph commands through `nix develop --command ...`.
- Always use Hetzner SFTP input and cached downloads; remove local and uncached modes.
- Derive `/home/ds_v2_metadata/<parquet-stem>_metadata.csv` exactly.
- Parquet owns audio bytes; CSV owns all metadata fields.
- Missing files, missing headers, row-count differences, and identity mismatches fail clearly without fallback.
- Retain at most four ds_v2 Parquet/CSV cache pairs and prune the least recently used pair.
- Do not commit permanent tests; use temporary checks and remove them before completion.
- Keep every modified or created file under 300 lines.

---

### Task 1: Strict cached Parquet/CSV row source

**Files:**
- Create: `src/runner/nodes/hetzner/ds_v2_rows.py`
- Temporary check: `/tmp/check_ds_v2_rows.py`

**Interfaces:**
- Produces: `DsV2Row(index: int, audio: bytes, metadata: dict[str, str])`.
- Produces: `metadata_remote_path(remote_parquet_path: str) -> str`.
- Produces: `load_rows(host: str, remote_parquet_path: str, cache_dir: Path, retries: int, row_offset: int, row_limit: int) -> list[DsV2Row]`.
- Consumes: PyArrow only for Parquet row count, identity columns, and selected audio bytes; standard-library CSV for metadata.

- [ ] **Step 1: Write a failing temporary check for exact path derivation and strict pair validation**

Create `/tmp/check_ds_v2_rows.py` with temporary Parquet/CSV fixtures. Assert:

```python
assert metadata_remote_path("/home/ds_v2/foo_processed.parquet") == (
    "/home/ds_v2_metadata/foo_processed_metadata.csv"
)
assert load_local_pair(valid_parquet, valid_csv, 0, 2)[0].metadata["text_src"] == "first"
```

Then create mismatched fixtures and assert errors contain these fragments:

```python
"missing required CSV columns"
"row count mismatch"
"identity mismatch at row 1 field speaker_id"
```

`load_local_pair` is an internal file-pair loader used by `load_rows` after caching, so the temporary check does not invoke a node directly.

- [ ] **Step 2: Run the temporary check and confirm the missing module/API failure**

Run:

```bash
nix develop --command python /tmp/check_ds_v2_rows.py
```

Expected: failure importing `runner.nodes.hetzner.ds_v2_rows` or the declared functions.

- [ ] **Step 3: Implement typed source paths, identities, and selected rows**

Create these structures in `ds_v2_rows.py`:

```python
@dataclass(frozen=True)
class DsV2RowIdentity:
    chunk_index: int
    sample_index: int
    sample_start: float
    speaker_id: str


@dataclass(frozen=True)
class DsV2Row:
    index: int
    audio: bytes
    metadata: dict[str, str]
```

Define `CSV_METADATA_COLUMNS` as every current ds_v2 metadata column except `audio`, and `IDENTITY_COLUMNS` as `chunk_index`, `sample_index`, `sample_start`, and `speaker_id`. `metadata_remote_path` must reject non-`.parquet` basenames and construct the exact metadata-directory filename.

- [ ] **Step 4: Implement mandatory reusable cache downloads**

Move the existing retrying SFTP download behavior into `ds_v2_rows.py`. `cached_remote_file` must always return a non-empty cache file keyed by the full remote path; otherwise download through a `.tmp` file and atomically replace the cache target. Its final error must include host, remote path, retry count, subprocess exit status, and captured SFTP output.

- [ ] **Step 5: Implement streaming full-pair validation**

`load_local_pair` must:

1. Open the Parquet schema and require `audio` plus all identity columns.
2. Open the CSV with `utf-8-sig`, require every `CSV_METADATA_COLUMNS` header, and reject duplicate headers.
3. Iterate all CSV rows against Parquet identity-only batches.
4. Parse both sides into `DsV2RowIdentity` and fail on the first differing field with file paths, absolute row index, field, and both values.
5. Fail when the total CSV and Parquet row counts differ.
6. Retain only requested CSV rows, then read only the selected Parquet audio range and return matching `DsV2Row` objects.

No Parquet metadata value may be copied into `DsV2Row.metadata`.

- [ ] **Step 6: Run the temporary strict-validation check**

Run:

```bash
nix develop --command python /tmp/check_ds_v2_rows.py
```

Expected: all path, valid-pair, missing-header, count-mismatch, and identity-mismatch assertions pass.

### Task 2: Integrate CSV-owned metadata into the runflow node

**Files:**
- Modify: `src/runner/nodes/hetzner/ds_v2_parquet.py`
- Modify: `workflows/ds_v2_sample_import.json`
- Modify: `workflows/README.md`

**Interfaces:**
- Consumes: `load_rows(...) -> list[DsV2Row]` from Task 1.
- Preserves: node type `HetznerDsV2ParquetAudioSource` and streaming `AudioPort` output.

- [ ] **Step 1: Remove obsolete source settings and transport code**

Delete `source`, `local_parquet_path`, and `cache_download` from `HetznerDsV2ParquetAudioSourceSettings`. Remove `_local_parquet_path`, `_cache_name`, `_download_sftp_file`, `_sftp_error_detail`, `_iter_parquet_rows`, and their unused imports/constants from `ds_v2_parquet.py`.

- [ ] **Step 2: Adapt node loading to validated rows**

Change `_load_audio_items` to call:

```python
rows = load_rows(
    host=settings.host,
    remote_parquet_path=settings.remote_parquet_path,
    cache_dir=Path(context.cache_dir) / "hetzner",
    retries=settings.download_retries,
    row_offset=settings.row_offset,
    row_limit=settings.row_limit,
)
```

Create voice IDs from each `DsV2Row.metadata`. For conversion, construct the row used by existing helpers as `{**item.metadata, "audio": item.audio}` and pass `item.index` as the absolute index. Do not restore any Parquet metadata fallback.

- [ ] **Step 3: Update discovery text and smoke workflow settings**

Update the node description to state that audio comes from cached Hetzner Parquet and metadata from the exact cached CSV pair. Remove `source`, `local_parquet_path`, and `cache_download` from `workflows/ds_v2_sample_import.json`. Update `workflows/README.md` with the mandatory exact-match convention and strict failure behavior.

- [ ] **Step 4: Check module size and static validity**

Run:

```bash
wc -l src/runner/nodes/hetzner/ds_v2_rows.py src/runner/nodes/hetzner/ds_v2_parquet.py
nix develop --command python -m compileall -q src/runner/nodes/hetzner
git diff --check
```

Expected: both Python files are at most 300 lines, compilation exits zero, and `git diff --check` prints nothing.

### Task 3: Verify the real Hetzner pair through a graph

**Files:**
- Temporarily modify and restore: `workflows/ds_v2_sample_import.json` only if environment-specific IDs require it.
- Remove: `/tmp/check_ds_v2_rows.py`

**Interfaces:**
- Consumes: the updated registered node and existing sample workflow.
- Produces: one saved audio record whose metadata came from the matched CSV.

- [ ] **Step 1: Restart the shared development stack**

Run:

```bash
nix develop --command runflow-dev-stop
nix develop --command runflow-dev-session
```

Detach the client while leaving the single `runflow-dev` Zellij session alive, then confirm with `nix develop --command runflow-dev-status`.

- [ ] **Step 2: Submit the existing ds_v2 graph through `POST /graphs/runs`**

Use `workflows/ds_v2_sample_import.json` with the known pair:

```text
/home/ds_v2/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet
/home/ds_v2_metadata/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed_metadata.csv
```

Expected: the graph is accepted and returns a run ID.

- [ ] **Step 3: Inspect the run using the project CLI**

Run:

```bash
nix develop --command python -m cli runs
nix develop --command python -m cli logs <run_id>
```

Expected: the source, record save, and segment save nodes complete. Inspect the saved result and confirm CSV values such as `speaker_id`, `text_src`, and nested source metadata are present.

- [ ] **Step 4: Exercise strict missing-pair failure through a temporary graph**

Submit a temporary copy whose remote Parquet basename has no corresponding metadata CSV. Inspect with:

```bash
nix develop --command python -m cli failed <run_id>
```

Expected: an actionable SFTP error names the exact derived `/home/ds_v2_metadata/*_metadata.csv` path; no Parquet metadata fallback occurs.

- [ ] **Step 5: Remove temporary checks and review scope**

Delete `/tmp/check_ds_v2_rows.py` and any temporary graph copy. Run `git status --short`, confirm no cache/model/audio artifacts are staged, and inspect the final diff only for the row source, importer, workflow, and README changes.
