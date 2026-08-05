# Common Voice Incremental Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents or git worktrees.

**Goal:** Convert the 30 verified Common Voice 26 archives into compliant Stage 1 datasets, append them to the three existing backend datasets, verify every stored field and audio byte, and prune verified local data.

**Architecture:** Extend the existing Common Voice preparer to discover locale archives per part and process them incrementally. Existing manifest records remain authoritative for already-uploaded source IDs; new records are speaker-balanced, normalized, merged by source ID, and uploaded through `imports/stage1_backend.py`. Each part is prepared, uploaded, reconciled, and pruned before the next part to keep disk use bounded.

**Tech Stack:** Python 3.12, Pydantic, tarfile, ffmpeg, soundfile, shared SQLAlchemy CRUD facades, S3-compatible packed audio storage, Nix, tmux.

## Global Constraints

- Run every Python and service command through `nix develop --command`.
- Run long preparation and import commands inside tmux.
- Audio must be 24 kHz, mono, PCM-24 WAV.
- Import no more than 50 hours per language and prefer speaker diversity.
- Preserve transcript, speaker identity, demographics, votes, locale, variant, segment, and original path metadata.
- Keep the three existing Common Voice part dataset identities and source IDs stable.
- Use shared database CRUD facades; do not issue ad hoc SQLAlchemy queries.
- Keep temporary extraction data empty after each archive and prune verified WAVs after backend reconciliation.
- Do not use subagents, worktrees, or committed tests.

---

### Task 1: Dynamic archive catalog and target allocation

**Files:**
- Create: `imports/stage1/common_voice_part1/src/catalog.py`
- Test temporarily: `imports/stage1/test_common_voice_prepare.py`

**Interfaces:**
- Produces: `ArchiveSpec(language: str, archive_locale: str, part: int, target_hours: float, path: Path)`
- Produces: `load_archive_specs(stage_root: Path) -> list[ArchiveSpec]`

- [ ] **Step 1: Write a failing test**

```python
def test_discovers_aliases_and_caps_targets(tmp_path):
    specs = load_archive_specs(tmp_path)
    assert [(item.language, item.archive_locale) for item in specs] == [
        ("ku", "kmr"),
        ("qu", "qxp"),
    ]
    assert all(item.target_hours <= 50.0 for item in specs)
```

- [ ] **Step 2: Run the test and confirm it fails because `catalog` does not exist**

Run: `nix develop --command python -m unittest imports.stage1.test_common_voice_prepare -v`

- [ ] **Step 3: Implement catalog discovery**

Parse each `common_voice_part*/tmp/*.tar.gz` first member as `cv-corpus-26.0-2026-06-12/<archive_locale>/...`, map `kmr` to `ku` and `qxp` to `qu`, and assign the verified waterfill allocation capped at 50 hours.

- [ ] **Step 4: Run the test and catalog audit**

Expected: 30 unique languages, 30 unique archives, every target in `(0, 50]`.

### Task 2: Incremental per-part preparation

**Files:**
- Modify: `imports/stage1/common_voice_part1/src/prepare.py`
- Test temporarily: `imports/stage1/test_common_voice_prepare.py`

**Interfaces:**
- Consumes: `load_archive_specs(stage_root)`
- Produces: `prepare_part(part: int) -> PreparationResult`
- Produces: merged `common_voice_partN/data.json` and `STATUS.md`

- [ ] **Step 1: Write failing tests**

```python
def test_merge_preserves_existing_and_replaces_matching_source_ids():
    merged = merge_records([old_a, old_b], [new_b, new_c])
    assert [item.source_id for item in merged] == ["a", "b", "c"]
    assert merged[1] == new_b

def test_metadata_retains_every_validated_tsv_field():
    record = convert_row(row)
    assert record.metadata["publisher_row"] == row
```

- [ ] **Step 2: Run tests and confirm the missing behavior fails**

Run: `nix develop --command python -m unittest imports.stage1.test_common_voice_prepare -v`

- [ ] **Step 3: Implement archive processing**

For each archive in the requested part: extract into a locale-specific temporary directory, load `validated.tsv` and `clip_durations.tsv`, speaker-balance to the catalog target, normalize selected MP3s with 16 workers, retain every TSV field under `publisher_row`, and remove the extraction directory.

- [ ] **Step 4: Implement manifest merge**

Load existing `data.json`, merge by stable `cv26:<language>:<filename>` source ID, retain old records whose WAVs were already backend-verified, include all old and new language limits, write atomically, and create `STATUS.md` with first line `COMPLETE`.

- [ ] **Step 5: Run tests and remove the temporary test module**

Expected: all tests pass; no test file remains in the repository.

### Task 3: Prepare and locally verify each part

**Files:**
- Runtime outputs: `imports/stage1/common_voice_partN/data.json`
- Runtime outputs: `imports/stage1/common_voice_partN/wavs/*.wav`
- Runtime logs: `imports/stage1/common_voice_partN/prepare-20260723.log`

**Interfaces:**
- Consumes: verified tar archives
- Produces: backend-ready manifests and WAVs

- [ ] **Step 1: Run part preparation in tmux**

Run one part at a time:

```bash
nix develop --command python -m imports.stage1.common_voice_part1.src.prepare --part N
```

- [ ] **Step 2: Verify new WAVs and manifests**

Check every locally present WAV with `soundfile`: 24,000 Hz, one channel, `PCM_24`; compare duration to the manifest; assert each new record has one dataset segment, transcript, stable speaker ID, voice prompt when demographics exist, and complete publisher metadata.

- [ ] **Step 3: Confirm limits and disk ceiling**

Aggregate new duration by language and assert each is at most 50 hours and within one clip of its target. Confirm filesystem use remains below 512 GB.

### Task 4: Upload, reconcile, and prune each part

**Files:**
- Runtime logs: `imports/stage1/common_voice_partN/upload-20260723.log`
- Runtime logs: `imports/stage1/common_voice_partN/verify-20260723.log`
- Runtime journal: `imports/stage1/common_voice_partN/.backend-verified-source-ids`

**Interfaces:**
- Consumes: merged manifest and new WAVs
- Produces: backend dataset containing exactly every manifest source ID

- [ ] **Step 1: Upload only absent source IDs**

```bash
nix develop --command python -m imports.stage1_backend import common_voice_partN
```

- [ ] **Step 2: Verify all backend fields and bytes, pruning verified WAVs**

Call `verify_stage_paths([data_path], prune_verified=True)` through Nix. Require exact source-ID equality, exact annotations/segments/prompts/metadata, and byte-for-byte stored audio equality.

- [ ] **Step 3: Confirm terminal state**

Assert the verification journal contains one line per manifest record, `wavs/` is empty, no extraction directory remains, and backend count equals manifest count.

- [ ] **Step 4: Remove verified tar archives for the part**

Only after the part passes reconciliation, remove its processed archives and stale download-session receipts; preserve the source code and final logs.

### Task 5: Final audit and tracker update

**Files:**
- Modify: `imports/dataset-download-groups-1000h.md`
- Modify: `imports/stage1-complete-slugs.txt`

**Interfaces:**
- Consumes: all three verified backend datasets
- Produces: terminal Common Voice import status

- [ ] **Step 1: Audit all three manifests against backend state**

Require exact record counts, per-language hours, source IDs, metadata, segment fields, speaker IDs, prompts, and stored audio bytes.

- [ ] **Step 2: Update tracker state**

Mark each Common Voice part complete only after its reconciliation succeeds and record it in the completed-slug registry only when cleanup invariants are satisfied.

- [ ] **Step 3: Check workspace and disk**

Confirm no temporary extraction files, partial downloads, unverified WAVs, or generated tests remain; confirm disk use stays below 512 GB.
