# Speaker Identity Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make string `speaker_id` the only persisted speaker identity and replace the Voices UI/API with a derived Speakers catalog.

**Architecture:** Shared speaker CRUD aggregates and mutates speaker IDs directly on audio rows and segment annotations. Backend `/speakers` exposes that facade; frontend Speakers consumes it. A direct migration removes the empty Voice table and `audio_files.voice_id`, while runner and shared types stop carrying Voice UUID identity.

**Tech Stack:** Python, SQLAlchemy, Alembic, FastAPI, Pydantic, React, TypeScript, TanStack Query

## Global Constraints

- Run Python, tests, and frontend commands through `nix develop --command`.
- Access PostgreSQL through shared CRUD facades.
- Keep `src/runflow` domain-agnostic.
- Do not retain compatibility aliases for `voice_id` or `/voices`.
- Do not commit temporary tests.

---

### Task 1: Shared identity model and migration

**Files:**
- Modify: `src/shared/audio_annotations.py`
- Modify: `src/shared/db/audio/models.py`
- Modify: `src/shared/db/audio/schemas.py`
- Modify: `src/shared/db/audio/crud.py`
- Modify: `src/shared/db/audio/pack_crud.py`
- Modify: `src/shared/db/audio/external_crud.py`
- Modify: `src/shared/db/audio/references_crud.py`
- Modify: `src/shared/db/audio/segment_references_crud.py`
- Modify: `src/shared/db/audio/speaker_assignment_crud.py`
- Delete: `src/shared/db/voices/`
- Create: `migrations/versions/20260721_01_remove_voice_identity.py`
- Test: `/tmp/test_speaker_identity_schema.py`

**Interfaces:**
- Produces: `AudioAnnotations.speaker_id: str | None` as the sole identity field.
- Produces: speaker assignment payloads carrying string `speaker_id`.

- [ ] Write a temporary failing schema test asserting `voice_id` is rejected and the audio table has no `voice_id` column.
- [ ] Run it through Nix and confirm failure on the existing field.
- [ ] Remove Voice UUID fields from shared types, persistence, and reference projections; add the direct Alembic drop migration.
- [ ] Run the test and Python compile checks to green.
- [ ] Commit the shared model and migration.

### Task 2: Derived speaker CRUD and API

**Files:**
- Modify: `src/shared/db/speakers/crud.py`
- Create: `src/shared/db/speakers/catalog_crud.py`
- Modify: `src/shared/db/speakers/schemas.py`
- Delete: `src/backend/voices/api.py`
- Create: `src/backend/speakers/api.py`
- Modify: `src/backend/api.py`
- Test: `/tmp/test_speaker_catalog.py`

**Interfaces:**
- Produces: `search_speakers(session, query, limit, offset) -> tuple[list[SpeakerRead], int]`.
- Produces: `rename_speaker(session, speaker_id, replacement)` and `clear_speaker(session, speaker_id)` that update audio and segment annotations atomically.
- Produces: `GET /speakers`, `PATCH /speakers/{speaker_id}`, and `DELETE /speakers/{speaker_id}`.

- [ ] Write a failing CRUD test using temporary audio rows with both audio-level and segment-level speaker annotations.
- [ ] Confirm aggregation, rename, and clear expectations fail before implementation.
- [ ] Implement aggregation and mutations in shared CRUD, then expose the new backend routes.
- [ ] Run the CRUD test and direct FastAPI endpoint smoke checks.
- [ ] Commit speaker catalog CRUD and API.

### Task 3: Runner identity port

**Files:**
- Modify: project-owned Python files under `src/runner/nodes/` returned by `rg '\bvoice_id\b'`.
- Modify: active workflow JSON files containing project identity `voice_id`.
- Test: `/tmp/test_no_runner_voice_identity.py`

**Interfaces:**
- Consumes: `AudioAnnotations.speaker_id` from Task 1.
- Produces: runner payloads, assignments, statistics, TTS inputs, and training records that use `speaker_id` only.

- [ ] Write a failing static test listing project-owned runner identity occurrences of `voice_id`.
- [ ] Classify external provider/model vocabulary separately from persisted project identity.
- [ ] Port persisted identity uses to string `speaker_id`, including clustering assignment and training manifest paths.
- [ ] Run compile checks and relevant graph/node schema smoke checks.
- [ ] Commit the runner port.

### Task 4: Rename frontend Voices to Speakers

**Files:**
- Rename: `src/frontend/src/features/voices/` to `src/frontend/src/features/speakers/`.
- Modify: `src/frontend/src/app/ScreenRouter.tsx`
- Modify: frontend navigation state/components referencing Voices.
- Modify: `src/frontend/src/features/audio/AudioToolbar.tsx`
- Modify: `src/frontend/src/features/audio/SegmentRow.tsx`
- Modify: `src/frontend/src/features/audio/actions.ts`
- Modify: frontend audio/testing types containing project `voice_id`.
- Test: `/tmp/test_frontend_speaker_identity.sh`

**Interfaces:**
- Consumes: `/speakers` API from Task 2.
- Produces: Speakers screen and speaker selectors sourced through TanStack Query.

- [ ] Write a failing static frontend test rejecting `/voices`, `SPEAKER_NAMES`, and project `voice_id`.
- [ ] Rename the feature and route, remove create, and connect list/rename/delete to `/speakers`.
- [ ] Replace static speaker selectors with query-backed speaker options.
- [ ] Run the static test, TypeScript check, and frontend build.
- [ ] Commit the frontend port.

### Task 5: End-to-end verification and cleanup

**Files:**
- Modify only defects exposed by verification.
- Delete all `/tmp/test_*speaker*` scripts created by this plan.

**Interfaces:**
- Verifies the complete speaker-only contract.

- [ ] Apply the migration through the managed dev stack and verify imported speaker totals remain unchanged.
- [ ] Call `/speakers` and verify imported speakers, counts, and datasets are visible.
- [ ] Exercise a reversible speaker rename through the API and verify both audio and segment annotations, then restore it.
- [ ] Run repository searches, Python compile checks, frontend build, and a real graph using speaker-bearing audio.
- [ ] Remove temporary tests and report any external-library uses of “voice” that intentionally remain.
- [ ] Commit verification fixes, if any.
