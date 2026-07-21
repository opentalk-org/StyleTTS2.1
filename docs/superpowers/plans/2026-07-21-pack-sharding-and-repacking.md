# Pack Sharding and Repacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store new audio and waveform data in 256 MiB packs under approximately 256-file folders, then safely repack every legacy audio pack and verify the complete backend corpus.

**Architecture:** A generic database-backed folder allocator selects an underfilled UUID folder for either pack model and tolerates small concurrent overshoot. Audio migration processes bounded legacy-pack groups, verifies replacement bytes before committing references, and deletes old objects only after commit.

**Tech Stack:** Python 3.12, SQLAlchemy, PostgreSQL, boto3-compatible object storage, FastAPI backend, Nix development shell, tmux

## Global Constraints

- Run every Python, pytest, backend, runner, and CLI command through `nix develop --command`.
- Use shared CRUD facades and `database_session`; do not add ad hoc persistence outside `src/shared/db`.
- Keep modified files below 300 lines and folders below 16 files.
- Keep temporary tests outside the committed tree and remove them after verification.
- Do not migrate legacy waveform packs.
- Do not delete legacy audio objects until replacement bytes and committed references are verified.
- Use the current checkout; do not create a branch, worktree, or subagent.

---

### Task 1: Generic pack-folder allocation

**Files:**
- Create: `src/shared/db/pack_folders.py`
- Temporary test: `/tmp/test_pack_folders.py`

**Interfaces:**
- Consumes: SQLAlchemy `Session`, a mapped model with a string `path` column, and a path prefix.
- Produces: `PackFolderAllocator(session, model, prefix, target_files=256)` with `path_for(pack_id: UUID) -> str`.

- [ ] **Step 1: Write a failing temporary test**

Create `/tmp/test_pack_folders.py` with `apply_patch`. Use temporary PostgreSQL rows for both `BucketFile` and `WaveformPack`. Assert that `path_for()` returns `<prefix>/<folder UUID>/<pack UUID>.bin`, reuses a folder below 256 registered files, rotates once the allocator's database count plus local allocations reaches 256, and reuses capacity after a row is deleted.

- [ ] **Step 2: Run the test and observe the missing-module failure**

Run:

```bash
nix develop --command python /tmp/test_pack_folders.py
```

Expected: `ModuleNotFoundError: shared.db.pack_folders`.

- [ ] **Step 3: Implement the allocator**

Create a focused module containing:

```python
@dataclass
class PackFolderAllocator:
    session: Session
    model: type[Any]
    prefix: str
    target_files: int = 256

    def path_for(self, pack_id: UUID) -> str:
        folder = self._folder_with_capacity()
        self._pending[folder] += 1
        return f"{self.prefix}/{folder}/{pack_id}.bin"
```

Initialize `_pending` in `__post_init__`. Count only paths matching `prefix/<folder>/<file>`; ignore legacy flat paths. Choose the least-populated folder with stable lexical tie-breaking, or create a UUID folder. Include pending allocations so one writer rotates locally at 256. Validate positive `target_files` explicitly.

- [ ] **Step 4: Run the temporary test**

Run the command from Step 2. Expected: all allocator assertions pass and temporary rows roll back.

- [ ] **Step 5: Commit the allocator**

```bash
git add src/shared/db/pack_folders.py
git commit -m "feat: allocate sharded pack folders"
```

### Task 2: Adopt 256 MiB sharded packs for audio and waveforms

**Files:**
- Modify: `src/shared/db/audio/pack_store.py`
- Modify: `src/shared/db/waveforms/pack_store.py`
- Temporary test: `/tmp/test_pack_writers.py`

**Interfaces:**
- Consumes: `PackFolderAllocator.path_for()` from Task 1.
- Produces: `AudioPackConfig(target_pack_bytes=256 * 1024 * 1024, folder_target_files=256)` and equivalent `WaveformPackConfig`.

- [ ] **Step 1: Write failing writer tests**

Create `/tmp/test_pack_writers.py`. With database rollback and an in-memory object store, exercise public audio and waveform bulk CRUD. Assert default target sizes are 256 MiB, created paths have two components below their prefix, 257 oversized writes rotate to a second folder, and CRUD readback equals input bytes.

- [ ] **Step 2: Run and confirm the old defaults/path fail**

```bash
nix develop --command python /tmp/test_pack_writers.py
```

Expected: failures show the 128 MiB audio default, 64 MiB waveform default, and flat paths.

- [ ] **Step 3: Update both writers minimally**

Add `folder_target_files: int = 256` to both frozen configuration dataclasses. Construct one `PackFolderAllocator` per writer and replace flat `_create_pack` paths with `allocator.path_for(pack_id)`. Generate the pack UUID before constructing its model. Keep oversized payload, staging, transaction, and read behavior unchanged.

- [ ] **Step 4: Run writer tests and focused project tests**

```bash
nix develop --command python /tmp/test_pack_writers.py
nix develop --command python -m pytest -q src/shared/db/audio src/shared/db/waveforms
```

Expected: temporary assertions pass; any discovered focused tests pass. If no collected tests exist, record that and rely on the temporary test plus Task 3 graph smoke.

- [ ] **Step 5: Commit writer changes**

```bash
git add src/shared/db/audio/pack_store.py src/shared/db/waveforms/pack_store.py
git commit -m "feat: write 256 MiB sharded media packs"
```

### Task 3: Validate new writes through real workflows

**Files:**
- Temporary graph JSON: `/tmp/pack-storage-smoke.json`
- No committed source changes.

**Interfaces:**
- Consumes: backend graph API, runner registry, audio/waveform public CRUD behavior.
- Produces: run IDs and read/update/delete evidence for the migration gate.

- [ ] **Step 1: Confirm the shared stack and schema**

```bash
nix develop --command runflow-dev-status
```

Inspect existing smoke workflows and testing feeder nodes. Build the smallest graph that persists audio and generates/reads waveform data through registered nodes; do not call node `execute()` directly.

- [ ] **Step 2: Submit and inspect the graph**

Submit `/tmp/pack-storage-smoke.json` through `POST /graphs/runs`. Inspect it with:

```bash
nix develop --command python -m cli runs
nix develop --command python -m cli logs <run_id>
nix develop --command python -m cli failed <run_id>
```

Expected: successful run with no failed node.

- [ ] **Step 3: Exercise iterative delete/add**

Through public CRUD in a temporary Nix Python script, record the created folder, delete the smoke audio and waveform rows, purge their orphan packs, then create replacements. Assert the replacement paths use valid sharded folders and readback bytes match.

- [ ] **Step 4: Confirm the migration gate**

List pack metadata through asset/waveform CRUD and physical objects through read-only rclone listing. Expected: no missing new objects or size mismatches. Do not begin migration unless Steps 2–4 pass.

### Task 4: Bounded, verified legacy-audio repacking

**Files:**
- Create: `src/shared/db/audio/repack_crud.py`
- Modify: `src/shared/db/audio/__init__.py`
- Create: `imports/repack_audio.py`
- Temporary test: `/tmp/test_audio_repack.py`

**Interfaces:**
- Produces: `repack_legacy_audio_packs(session, store=None, max_source_bytes=512 * 1024 * 1024) -> RepackResult` and a CLI that loops until `remaining_packs == 0`.
- `RepackResult` contains moved audio count, replaced pack count, created pack count, bytes verified, deleted paths, and remaining legacy count.

- [ ] **Step 1: Write a failing migration test**

Create legacy flat packs containing multiple audio rows in a rollback-scoped database and in-memory store. Include a partial pack and an oversized audio file. Assert one bounded call preserves every byte and metadata field, creates only sharded paths, removes only replaced legacy objects, and a second call resumes remaining work.

- [ ] **Step 2: Run and confirm the missing API failure**

```bash
nix develop --command python /tmp/test_audio_repack.py
```

Expected: import failure for `repack_legacy_audio_packs`.

- [ ] **Step 3: Implement one migration group**

Select legacy `audio-packs/<file>.bin` rows with `FOR UPDATE`, accumulating packs until the bounded source-byte limit. Load their audio rows with locks and download only selected source packs. Hash every source slice. Write those exact slices through `AudioPackWriter`, update references and offsets, flush replacements, and download replacement packs before commit to compare every slice hash. Raise on any mismatch.

After successful verification, delete legacy pack rows and commit. Only then delete their objects. Return exact counters. Resolve storage through the normal settings facade when no store is supplied.

- [ ] **Step 4: Implement the resumable CLI**

`imports/repack_audio.py` opens a fresh `database_session` for each group, prints tqdm progress in packs/bytes, checks cancellation through SIGINT between groups, and exits nonzero on any failed verification. It must not query or mutate audio metadata outside the public repack CRUD function.

- [ ] **Step 5: Run migration tests and compilation**

```bash
nix develop --command python /tmp/test_audio_repack.py
nix develop --command python -m compileall -q src/shared/db/audio imports/repack_audio.py
```

Expected: byte/metadata/resume assertions pass and compilation exits zero.

- [ ] **Step 6: Commit migration code**

```bash
git add src/shared/db/audio/repack_crud.py src/shared/db/audio/__init__.py imports/repack_audio.py
git commit -m "feat: repack legacy audio storage safely"
```

### Task 5: Repack production data and verify everything

**Files:**
- No committed source changes.
- Runtime logs: `/tmp/repack-audio.log`, `/tmp/repack-audio.exit`.

**Interfaces:**
- Consumes: validated migration CLI from Task 4 and the existing 395,767-row backend corpus.
- Produces: fully sharded 256 MiB audio storage and final integrity report.

- [ ] **Step 1: Capture the pre-migration inventory**

Through CRUD, record audio count, dataset membership counts, total duration, metadata/segment digests, pack count/bytes, and legacy/sharded path counts. Use read-only rclone listing to record physical object count, missing objects, size mismatches, and the 20 known pre-existing orphans.

- [ ] **Step 2: Run repacking in tmux**

```bash
tmux new-session -d -s audio-repack \
  "nix develop --command runuser -u user -- env HOME=/home/user \
  RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow \
  PYTHONPATH=src python imports/repack_audio.py > /tmp/repack-audio.log 2>&1; \
  echo \$? > /tmp/repack-audio.exit"
```

Monitor progress, memory, backend health, and R3 connectivity. Stop before destructive cleanup if replacement verification fails.

- [ ] **Step 3: Compare post-migration invariants**

Require exact equality for audio row count, per-dataset counts, total duration, metadata/segment digests, and summed audio byte lengths. Require zero legacy registered pack paths, zero missing objects, zero size mismatches, and successful CRUD readback samples from every dataset.

- [ ] **Step 4: Run real graph smoke again**

Repeat Task 3's graph submission and CLI inspection. Expected: successful audio and waveform workflow with no failed node after migration.

- [ ] **Step 5: Report layout and clean temporary tests**

Report live pack count, physical object count, folder count, min/median/max folder population, pack-size distribution, utilization, and remaining physical orphans. Remove temporary scripts and graph files through a recoverable or explicitly targeted operation. Do not remove diagnostic logs until the final report is delivered.

