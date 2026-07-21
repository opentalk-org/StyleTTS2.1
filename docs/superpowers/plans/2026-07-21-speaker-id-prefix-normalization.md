# Speaker ID Prefix Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Namespace CREMA-D and EmoDB speaker IDs in future staged imports and all existing backend metadata.

**Architecture:** Dataset preparation owns speaker-ID qualification; the shared uploader remains a transparent consumer. Existing rows are renamed through `shared.db.speakers.crud.rename_speaker`, which updates audio-file columns and embedded segment annotations atomically per speaker.

**Tech Stack:** Python, Pydantic stage manifests, SQLAlchemy shared CRUD, PostgreSQL, Nix development shell.

## Global Constraints

- CREMA-D IDs use `crema_d_<source-id>`.
- EmoDB IDs use `emodb_<source-id>`.
- Other datasets, audio bytes, packs, waveforms, and empty segments remain unchanged.
- All project commands run through `nix develop --command`.
- Temporary tests are removed before completion.

---

### Task 1: Dataset preparation qualification

**Files:**
- Modify: `imports/stage1/common/transcribed_parquet.py`
- Modify: `imports/stage1/crema_d/src/prepare.py`
- Modify: `imports/stage1/jl_corpus/src/prepare.py`
- Modify: `imports/stage1/emodb/src/prepare.py`
- Test: `/tmp/test_speaker_prefixes.py`

**Interfaces:**
- Consumes: source `row["speaker_id"]` values and EmoDB filename speaker codes.
- Produces: `DatasetConfig.speaker_prefix: str | None` and qualified staged `speaker_id` strings.

- [ ] **Step 1: Write the failing test**

Create a temporary test that constructs a CREMA-D `ShardTask`, calls `build_record`, and asserts `crema_d_1005`; inspect the EmoDB record expression and assert it contains the `emodb_` namespace.

- [ ] **Step 2: Verify the test fails**

Run `nix develop --command python /tmp/test_speaker_prefixes.py` and expect the CREMA-D assertion to report `1005 != crema_d_1005`.

- [ ] **Step 3: Implement qualification**

Add a required `speaker_prefix: str | None` field to `DatasetConfig`. In `build_record`, emit `f"{task.config.speaker_prefix}_{row['speaker_id']}"` when configured and preserve the source ID otherwise. Configure CREMA-D with `"crema_d"`, JL Corpus with `None`, and emit `f"emodb_{Path(row['audio']['path']).stem[:2]}"` in EmoDB.

- [ ] **Step 4: Verify the test passes**

Run `nix develop --command python /tmp/test_speaker_prefixes.py` and expect `PASS`.

### Task 2: Current staged manifests

**Files:**
- Modify: `imports/stage1/crema_d/data.json`
- Modify: `imports/stage1/emodb/data.json`
- Test: `/tmp/normalize_speaker_manifests.py`

**Interfaces:**
- Consumes: existing JSON manifests.
- Produces: manifests whose audio-level speaker IDs are qualified while source metadata remains untouched.

- [ ] **Step 1: Write and dry-run the manifest transformer**

Create a temporary script that loads each manifest, asserts every current audio-level ID is numeric, prefixes it, verifies 7,442 CREMA-D and 535 EmoDB records, and prints the replacement counts without writing.

- [ ] **Step 2: Run the dry-run**

Run `nix develop --command python /tmp/normalize_speaker_manifests.py --check` and expect `CREMA-D=7442 EmoDB=535`.

- [ ] **Step 3: Apply the manifest transformation**

Run `nix develop --command python /tmp/normalize_speaker_manifests.py --write`; use the script only as a mechanical formatter for these generated manifests.

- [ ] **Step 4: Verify staged values**

Run the check again and assert all CREMA-D IDs match `crema_d_[0-9]+`, all EmoDB IDs match `emodb_[0-9]+`, and source-row metadata retains the original IDs.

### Task 3: Existing backend metadata

**Files:**
- Test/migration: `/tmp/rename_live_speakers.py`

**Interfaces:**
- Consumes: the public `speaker_crud.search_speakers` and `speaker_crud.rename_speaker` functions.
- Produces: qualified audio-file and segment-annotation speaker IDs in PostgreSQL.

- [ ] **Step 1: Capture and validate the live rename set**

Use dataset membership from speaker catalog rows to build exactly 91 CREMA-D and 10 EmoDB numeric rename pairs. Capture summed audio-file and segment counts and reject any replacement already present.

- [ ] **Step 2: Apply renames through shared CRUD**

For each pair, open `database_session()` and call `speaker_crud.rename_speaker(session, old_id, replacement)`. Do not issue ad hoc SQL or edit pack objects.

- [ ] **Step 3: Verify live invariants**

Re-query through CRUD and assert zero numeric CREMA-D/EmoDB IDs, exactly 101 expected replacements, unchanged audio-file and segment totals, and zero IDs assigned to multiple datasets.

- [ ] **Step 4: Run repository checks and clean temporary files**

Run `nix develop --command python -m compileall -q imports/stage1/common imports/stage1/crema_d/src imports/stage1/jl_corpus/src imports/stage1/emodb/src`, inspect `git diff --check`, and delete all `/tmp` scripts.
