# MOS History Edit and Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow every MOS comparison to be changed or literally deleted without browser-native confirmation dialogs while keeping audio scores consistent with remaining history.

**Architecture:** Keep the existing comparison table and PATCH/DELETE routes. Mutation helpers update or remove any comparison, rewire the next comparison's `previous_score` values, and update current audio scores only when the mutated comparison is the newest one involving that audio. The frontend retains inline editing and replaces Undo with immediate Delete.

**Tech Stack:** Python 3.12, SQLAlchemy, FastAPI, React, TypeScript, TanStack Query, unittest, Nix development shell.

## Global Constraints

- Run every Python, frontend, and test command through `nix develop --command ...`.
- Keep temporary tests under `.tmp_mos_tests/` and remove them before completion.
- Preserve all dependency-override comments and do not edit dependency override code.
- Do not modify `src/runflow`; this behavior belongs to shared MOS persistence, backend MOS API, and frontend MOS history.
- Keep every changed file under 300 lines and every touched folder at 16 files or fewer.
- Preserve unrelated staged and working-tree changes.

---

### Task 1: Permit arbitrary comparison mutations

**Files:**
- Create temporarily: `.tmp_mos_tests/test_mos_history_mutations.py`
- Modify: `src/shared/db/mos/crud.py`

**Interfaces:**
- Consumes: `MosComparison`, `AudioFile`, `MosRatingUpdate`, and an SQLAlchemy `Session`.
- Produces: `update_rating(session, comparison_id, payload) -> MosComparison` and `delete_rating(session, comparison_id) -> None`.

- [ ] **Step 1: Write failing integration tests**

Create two ordered comparisons involving the same two audio rows in PostgreSQL. Assert:

```python
updated = mos_crud.update_rating(session, older.id, update_payload)
self.assertEqual(updated.score_a, 3.0)
self.assertEqual(audio_a.score, newer.score_a)
self.assertEqual(newer.previous_score_a, 3.0)

mos_crud.delete_rating(session, older.id)
self.assertEqual(audio_a.score, newer.score_a)
self.assertEqual(newer.previous_score_a, original_score_a)

mos_crud.delete_rating(session, newer.id)
self.assertEqual(audio_a.score, original_score_a)
```

Use unique UUIDs and explicit `created_at` values. Delete the temporary comparisons, audio rows, dataset, and bucket row in `tearDown`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
nix develop --command env RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow \
  python -m unittest discover -s .tmp_mos_tests -p 'test_mos_history_mutations.py'
```

Expected: FAIL because `update_rating` and `delete_rating` are not exposed and older comparisons are rejected.

- [ ] **Step 3: Implement ordered score-chain helpers**

Add a strict comparison lookup plus helpers that locate the immediately previous and next comparisons involving one audio, ordered by `(created_at, id)`. Read or write the correct `score_a`/`score_b` and `previous_score_a`/`previous_score_b` field according to which side contains the audio ID.

`update_rating` must:

```python
comparison = _comparison(session, comparison_id)
_validate_preferred(comparison, payload.preferred_audio_id)
for audio_id, new_score in _comparison_scores(comparison, payload.score_a, payload.score_b):
    next_comparison = _next_audio_comparison(session, comparison, audio_id)
    if next_comparison is None:
        _set_current_audio_score(session, audio_id, new_score, updated_at)
    else:
        _set_previous_score(next_comparison, audio_id, new_score)
```

`delete_rating` must rewire each next comparison to the deleted comparison's previous score. When no next comparison exists, set the audio's current score to the previous remaining comparison's score or the deleted comparison's stored previous score. Commit all changes once.

- [ ] **Step 4: Run the mutation tests and verify GREEN**

Run the Step 2 command. Expected: all mutation tests pass.

- [ ] **Step 5: Commit persistence behavior**

```bash
git add src/shared/db/mos/crud.py
git commit --only src/shared/db/mos/crud.py -m "feat: edit and delete any MOS comparison"
```

Do not add `.tmp_mos_tests/`.

---

### Task 2: Expose every comparison as editable and deletable

**Files:**
- Modify: `src/backend/mos/api.py`
- Modify temporarily: `.tmp_mos_tests/test_mos_history_mutations.py`

**Interfaces:**
- Consumes: `mos_crud.update_rating` and `mos_crud.delete_rating`.
- Produces: unchanged PATCH and DELETE HTTP paths; every history row returns `can_modify=true`.

- [ ] **Step 1: Add failing API contract assertions**

Assert the API source calls `update_rating` and `delete_rating`, history construction does not compare rows to a latest ID, and `_rating_response(..., True)` is used for every row.

- [ ] **Step 2: Run the tests and verify RED**

Run the Task 1 test command. Expected: FAIL because the API still calls newest-only CRUD functions and marks older rows non-modifiable.

- [ ] **Step 3: Update the MOS API wiring**

Remove the latest-ID lookup from list responses, pass `True` for every row, call `update_rating` from PATCH, and call `delete_rating` from DELETE. Rename the delete route handler from `undo_mos_rating` to `delete_mos_rating` without changing its URL or status code.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 test command. Expected: all tests pass.

- [ ] **Step 5: Commit API behavior**

```bash
git add src/backend/mos/api.py
git commit --only src/backend/mos/api.py -m "feat: expose all MOS history mutations"
```

---

### Task 3: Replace Undo with immediate Delete

**Files:**
- Modify: `src/frontend/src/features/mos/api.ts`
- Modify: `src/frontend/src/features/mos/query.ts`
- Modify: `src/frontend/src/features/mos/MosHistoryList.tsx`
- Modify: `src/frontend/src/features/mos/MosHistoryRow.tsx`
- Create temporarily: `.tmp_mos_tests/test_mos_history_ui.py`

**Interfaces:**
- Produces: `deleteMosRating(id: string) -> Promise<void>` and a `Delete` action on every row.

- [ ] **Step 1: Write failing source contract tests**

Assert that the four MOS frontend files contain no `window.confirm`, `window.alert`, or `window.prompt`; `MosHistoryRow.tsx` contains `Delete` and no `Undo`; and the API/query layer exports and uses `deleteMosRating` rather than `undoMosRating`.

- [ ] **Step 2: Run the UI tests and verify RED**

```bash
nix develop --command python -m unittest discover -s .tmp_mos_tests -p 'test_mos_history_ui.py'
```

Expected: FAIL because the row still uses `window.confirm`, `Undo`, and undo-named functions.

- [ ] **Step 3: Implement the immediate Delete action**

Rename undo-related functions and props to delete-related names. Remove the native confirmation branch. The handler must call delete directly, show `MOS comparison deleted` on success, show `Could not delete MOS comparison` on failure, and retain mutation pending-state disabling.

- [ ] **Step 4: Run tests and the frontend build**

```bash
nix develop --command python -m unittest discover -s .tmp_mos_tests -p 'test_mos_history_*.py'
nix develop --command bash -lc 'cd src/frontend && npm run build'
```

Expected: all tests pass and Vite completes successfully.

- [ ] **Step 5: Commit the frontend behavior**

```bash
git add src/frontend/src/features/mos/api.ts src/frontend/src/features/mos/query.ts \
  src/frontend/src/features/mos/MosHistoryList.tsx src/frontend/src/features/mos/MosHistoryRow.tsx
git commit --only src/frontend/src/features/mos/api.ts src/frontend/src/features/mos/query.ts \
  src/frontend/src/features/mos/MosHistoryList.tsx src/frontend/src/features/mos/MosHistoryRow.tsx \
  -m "feat: delete any MOS comparison"
```

---

### Task 4: Live verification and cleanup

**Files:**
- Delete: `.tmp_mos_tests/test_mos_history_mutations.py`
- Delete: `.tmp_mos_tests/test_mos_history_ui.py`

- [ ] **Step 1: Run a live API smoke flow**

Through the shared non-root development session, create a throwaway dataset with two audio files and three comparisons. PATCH the oldest comparison, DELETE the middle comparison, verify all returned rows have `can_modify=true`, and verify current audio scores still reflect the newest remaining comparison. Delete the newest and oldest comparisons and verify the pre-MOS scores are restored.

- [ ] **Step 2: Run final verification**

```bash
nix develop --command python -m compileall -q src
nix develop --command bash -lc 'cd src/frontend && npm run build'
nix develop --command python -c 'from backend.api import app; assert "/mos/ratings" in app.openapi()["paths"]'
git diff --check
```

Expected: every command exits zero. Confirm every changed file is at most 300 lines and every touched folder contains at most 16 files.

- [ ] **Step 3: Remove temporary tests and smoke data**

Delete `.tmp_mos_tests/` through a patch, remove generated caches, and delete the throwaway audio/dataset rows through public APIs. Confirm `git status --short` shows only committed MOS changes plus the user's unrelated pre-existing changes.
