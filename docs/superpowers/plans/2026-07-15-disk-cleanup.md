# Disk Cleanup Implementation Plan

> **For agentic workers:** Execute inline in the current checkout. Do not use subagents or create a worktree.

**Goal:** Reclaim stale object data, Rust build outputs, duplicate Hetzner cache files, and root-owned caches without deleting source repositories, database-backed checkpoints, or Codex/Claude conversation history.

**Expected reclaim:** Approximately 90 GB.

## Global constraints

- Run project and database commands through `nix develop --command ...`.
- Do not delete RustFS checkpoints, extra files, PostgreSQL, NATS, source trees, virtual environments, or model directories.
- Preserve `/root/.codex/sessions`, `/root/.codex/memories`, `/root/.claude/projects`, and `/root/.claude/file-history`.
- Record `df -h /`, category sizes, object counts, and database counts before and after cleanup.
- Stop if any audio-file or waveform rows still reference a pack selected for deletion.

### Task 1: Preflight invariants

- [x] Confirm there are no active graph runs and no pack-writing nodes executing.
- [x] Through shared database CRUD facades, confirm `audio_files` and `audio_waveforms` contain zero rows.
- [x] Enumerate audio-pack and waveform-pack database rows and verify none have live child references.
- [x] Record baseline disk usage and the exact object paths selected for deletion.

### Task 2: Purge stale audio and waveform packs

**Remove:**

- RustFS `audio-packs/` objects: approximately 40 GB across 284 objects.
- RustFS `waveform-packs/` objects: approximately 6.1 GB across 46 objects.
- Corresponding orphaned `bucket_files` and `waveform_packs` database rows.

- [x] Add focused public CRUD operations under `src/shared/db/audio/` and `src/shared/db/waveforms/` that select only packs with no live child rows, commit metadata deletion, then delete their object-store paths.
- [x] Exercise the operations through `database_session` and the configured `S3ObjectStore`; do not delete RustFS internal `part.1` paths directly.
- [x] Verify both object prefixes and their orphan pack rows are empty afterward.

### Task 3: Remove the duplicated ds_v2 cache pair

**Remove both rebuildable copies, reclaiming approximately 4.68 GiB:**

- `cache/hetzner/ds_v2_a3c13ae714734cc1_000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet`
- `cache/hetzner/a3c13ae714734cc1_000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet`

- [x] Verify their sizes and content hashes match before deletion.
- [x] Delete only these two cache files; leave other Hetzner Parquets and metadata caches intact.

### Task 4: Remove Rust build targets

**Remove approximately 34 GB:**

- `/workspace/tinfer/tinfer_rust/target` — 25 GB.
- `/workspace/tinfer/tinfer/espeak_align/target` — 287 MB.
- `/workspace/tinfer/tinfer_rust/espeak_align/target` — 148 MB.
- `/workspace/tinfer-worktrees/pysbd-scheduler-parity/tinfer_rust/target` — 3.2 GB.
- `/workspace/tinfer-worktrees/tinfer-rust-port/tinfer_rust/target` — 5.3 GB.

- [x] Confirm no Cargo build is running.
- [x] Remove only directories named `target`; preserve repositories, worktrees, source files, and lockfiles. The active `tinfer_rust` service subsequently rebuilt a minimal 588 MB release target.

### Task 5: Remove root caches and agent temporary data

**Remove approximately 6.2 GB:**

- Contents of `/root/.cache/` — 5.6 GB, primarily VS Code C++ indexing, Torch, and Nix evaluation caches.
- `/root/.codex/.tmp/` and `/root/.codex/tmp/` — approximately 77 MB.
- `/root/.claude/cache/`, `/root/.claude/jobs/`, and `/root/.claude/paste-cache/` — approximately 549 MB.

- [x] Confirm no active Codex or Claude process is using a selected temporary job directory.
- [x] Preserve Codex sessions/memories and Claude projects/file-history.
- [x] Remove cache contents while retaining parent directories where applications expect them. Preserve the active 4 KiB Codex lock directory.

### Task 6: Verify reclamation

- [x] Run `df -hT` and confirm the overlay is no longer critically full.
- [x] Re-run category-level `du` and confirm approximately 90 GB was reclaimed.
- [x] Confirm the shared runflow stack is healthy and the runner process is alive.
- [x] Confirm Codex and Claude conversation directories remain present and unchanged in file count.
- [x] Report actual reclaimed space and any planned item skipped due to a failed invariant.
