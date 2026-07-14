# Unified ds_v2 Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicate ds_v2 source and audio-record save nodes with settings-driven nodes that handle metadata-only references or real audio bytes in batches.

**Architecture:** `HetznerDsV2Source` always selects globally ordered metadata rows and conditionally resolves their inferred Parquet bytes. `SaveAudioRecord` dispatches each homogeneous micro-batch to either packed-audio creation or external-reference creation while preserving one typed port contract.

**Tech Stack:** Python 3.12, Pydantic, PyArrow Parquet, SQLAlchemy/PostgreSQL, runflow node runtime, Nix development shell.

## Global Constraints

- Run Python, backend, runner, and CLI commands through `nix develop --command ...`.
- Work in the current checkout; do not create a worktree or branch.
- Keep every file under 300 lines and every folder under 16 files.
- Apply offsets and limits globally across sorted metadata CSV rows.
- Never fall back to Parquet metadata.
- Use bulk CRUD functions for every selected save mode.
- Validate nodes through real graph submissions, never direct `execute()` calls.
- Do not retain temporary tests or generated run artifacts.

---

### Task 1: Metadata-driven selected-row audio loading

**Files:**
- Modify: `src/runner/nodes/hetzner/ds_v2_rows.py`
- Modify: `src/runner/nodes/hetzner/ds_v2_metadata_rows.py`
- Temporary test: `/tmp/test_ds_v2_selected_rows.py`

**Interfaces:**
- Consumes: `DsV2MetadataRow(index, remote_metadata_path, remote_parquet_path, metadata)`.
- Produces: `load_selected_audio_rows(host: str, rows: list[DsV2MetadataRow], cache_dir: Path, retries: int) -> list[DsV2Row]` in input order.

- [ ] **Step 1: Write a failing temporary test**

Build two small local Parquet/CSV fixtures, select non-contiguous
`DsV2MetadataRow` values spanning both Parquet files, and assert that
`load_selected_local_audio_rows(...)` returns the same order, bytes, and source
indices. Add assertions that a mismatched identity and an out-of-range source
index raise `ValueError` containing the Parquet path and row index.

- [ ] **Step 2: Run the test and verify the API is missing**

Run `nix develop --command python /tmp/test_ds_v2_selected_rows.py`.
Expected: import failure for `load_selected_local_audio_rows`.

- [ ] **Step 3: Implement grouped selected-row extraction**

Add a focused loader that groups selections by `(remote_metadata_path,
remote_parquet_path)`, caches each remote Parquet once, reads only the selected
indices, validates `chunk_index`, `sample_index`, `sample_start`, and
`speaker_id` against the already-selected CSV metadata, and restores original
selection order. Return `DsV2Row(index=row.index, audio=bytes,
metadata=row.metadata)`.

- [ ] **Step 4: Run the temporary test**

Run `nix develop --command python /tmp/test_ds_v2_selected_rows.py`.
Expected: all ordering, byte, identity, and range assertions pass.

### Task 2: Unified ds_v2 source node

**Files:**
- Modify: `src/runner/nodes/hetzner/ds_v2_metadata.py`
- Delete: `src/runner/nodes/hetzner/ds_v2_parquet.py`
- Modify: `src/runner/nodes/hetzner/__init__.py`
- Modify: `src/runner/nodes/registry.py`

**Interfaces:**
- Produces: registered `HetznerDsV2Source` with streaming `audio: AudioPort`.
- Settings: `host`, `row_offset`, `row_limit`, `import_audio`, `text_column`, `name_prefix`, `download_retries`, and `create_voices`.

- [ ] **Step 1: Rename and generalize the metadata source**

Rename its settings and node classes to `HetznerDsV2SourceSettings` and
`HetznerDsV2SourceNode`, set `NODE_TYPE = "HetznerDsV2Source"`, add
`import_audio: bool = False`, and retain the existing global metadata iterator.

- [ ] **Step 2: Attach bytes only in import mode**

For every queue-sized selection, call `load_selected_audio_rows(...)` only when
`import_audio` is true. Convert returned rows with `audio_from_row`; otherwise
retain `audio_metadata_from_row`. Voice creation remains one bulk operation per
selected queue chunk.

- [ ] **Step 3: Remove duplicate registration**

Delete the fixed-Parquet node and export/register only
`HetznerDsV2SourceNode`. Run
`nix develop --command python -m compileall -q src/runner/nodes/hetzner` and
expect exit zero.

### Task 3: Unified audio-record persistence node

**Files:**
- Modify: `src/runner/nodes/audio_segments/writeback.py`
- Delete: `src/runner/nodes/audio_segments/external_writeback.py`
- Modify: `src/runner/nodes/registry.py`
- Temporary test: `/tmp/test_save_audio_modes.py`

**Interfaces:**
- Settings: `storage_mode: Literal["stored", "external"] = "stored"`, `virtual`, and `bulk_import_packs`.
- Stored batches consume `Audio.data: bytes` and call `audio_crud.bulk_create_audio_files` once.
- External batches consume metadata-only audio and call `bulk_create_external_audio_files` once.

- [ ] **Step 1: Write failing payload-helper tests**

Create representative byte-backed and metadata-only `Audio` values. Assert the
stored payload contains bytes and the external payload contains provider, host,
Parquet path, and item index. Assert the wrong data shape for either mode raises
an error naming the audio ID and required mode.

- [ ] **Step 2: Run the test and verify external mode is absent**

Run `nix develop --command python /tmp/test_save_audio_modes.py`.
Expected: import or attribute failure for the unified external payload helper.

- [ ] **Step 3: Add storage-mode batch dispatch**

Move external payload and output conversion into a focused helper module if
needed to keep `writeback.py` under 300 lines. In `stored` mode build all
`AudioCreate` payloads and call packed CRUD once. In `external` mode build all
`ExternalAudioCreate` payloads and call external CRUD once. Preserve input
ordering and emit one `audio` plus `save_result` per input.

- [ ] **Step 4: Remove the separate external node and pass tests**

Delete its registration and module, then run
`nix develop --command python /tmp/test_save_audio_modes.py`.
Expected: all mode, payload, invariant, and ordering assertions pass.

### Task 4: Smoke workflows and real graph verification

**Files:**
- Modify: `workflows/ds_v2_metadata_import.json`
- Modify: `workflows/ds_v2_sample_import.json`
- Remove: `/tmp/test_ds_v2_selected_rows.py`
- Remove: `/tmp/test_save_audio_modes.py`

**Interfaces:**
- Metadata workflow pairs `import_audio=false` with `storage_mode="external"`.
- Sample workflow pairs `import_audio=true` with `storage_mode="stored"`.

- [ ] **Step 1: Update both workflows**

Use `HetznerDsV2Source`, remove `remote_parquet_path`, add `import_audio`, and
replace `SaveExternalAudioRecord` with `SaveAudioRecord` configured for the
matching mode.

- [ ] **Step 2: Check schema and source hygiene**

Run `rg -n 'HetznerDsV2MetadataSource|HetznerDsV2ParquetAudioSource|SaveExternalAudioRecord' src workflows` and expect no matches. Run `wc -l` on every
modified Python file and expect at most 300 lines. Run `git diff --check` and
expect no output.

- [ ] **Step 3: Restart the shared stack and submit small graphs**

Restart only through `nix develop --command runflow-dev-stop` and
`nix develop --command runflow-dev-session`. Submit both workflows through
`POST /graphs/runs` with a small non-destructive limit and inspect run state and
logs using `nix develop --command python -m cli runs` and
`nix develop --command python -m cli logs <run_id>`.

- [ ] **Step 4: Verify persisted results**

Confirm metadata mode stores external references without bucket bytes and audio
mode stores readable bytes. Confirm selected records reflect the configured
global row offset and limit.

- [ ] **Step 5: Remove temporary tests and review**

Remove both `/tmp` tests. Run compileall, workflow JSON parsing, `git diff
--check`, and `git status --short`; inspect the complete diff for only approved
source, persistence, registry, workflow, design, and plan changes.
