# Generic Workflow Reviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace speaker-audit report files with a generic, database-backed review drawer in Jobs whose approval launches a safe continuation graph.

**Architecture:** Runner nodes publish bounded typed `WorkflowReview` records linked to their producer job. The backend exposes generic review reads and row-locked decisions, and approval submits the stored continuation through `BackendManager`. The existing Jobs UI renders a domain-agnostic review drawer and reuses ranged audio endpoints through a clip-aware shared player.

**Tech Stack:** Pydantic, SQLAlchemy/PostgreSQL JSONB, Alembic, FastAPI, React/TypeScript, TanStack Query, Tailwind v4, Zustand navigation, runflow nodes.

## Global Constraints

- Keep `src/runflow` domain-agnostic; no scheduler special cases.
- Use shared CRUD facades for all database access.
- Keep review payloads bounded and Pydantic-validated.
- Keep every file under 300 lines and every folder under 16 files.
- Run every command through `nix develop --command ...`.
- Keep tests temporary under `tmp_tests/` and remove only the task-owned tests before each commit.
- Validate runner behavior through real graphs, never direct node `execute()` calls.

---

### Task 1: Generic review persistence

**Files:**
- Create: `src/shared/db/reviews/models.py`
- Create: `src/shared/db/reviews/schemas.py`
- Create: `src/shared/db/reviews/crud.py`
- Create: `src/shared/db/reviews/__init__.py`
- Create: `migrations/versions/20260714_05_add_workflow_reviews.py`
- Modify: `src/shared/db/models.py`
- Modify: `src/shared/db/jobs/crud.py`
- Modify: `src/shared/db/jobs/schemas.py`
- Test temporarily: `tmp_tests/test_workflow_reviews.py`

**Interfaces:**
- Produces `ReviewPayload`, `ReviewContinuation`, `ReviewCreate`, `ReviewRead`, `ReviewSummary`, `ReviewDecision`, and `ReviewState`.
- Produces CRUD `create_review`, `list_reviews_for_run`, `get_review`, and `decide_review`.

- [ ] **Step 1: Write the failing persistence tests**

Cover Pydantic rejection of unbounded/invalid media, idempotent `(kind, source_key)` creation, paginated run reads, same-decision idempotency, conflicting-decision failure, and deterministic continuation ID.

```python
created = review_crud.create_review(session, payload)
duplicate = review_crud.create_review(session, payload)
assert duplicate.id == created.id
approved = review_crud.decide_review(session, created.id, ReviewDecision.APPROVED)
assert approved.continuation_run_id == f"review_{created.id.hex}"
with pytest.raises(ValueError, match="already approved"):
    review_crud.decide_review(session, created.id, ReviewDecision.REJECTED)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `nix develop --command uv run --with pytest python -m pytest -q tmp_tests/test_workflow_reviews.py`

Expected: import failure for `shared.db.reviews`.

- [ ] **Step 3: Implement typed schemas and CRUD**

Use discriminated media and immutable bounded payloads:

```python
class AudioSegmentReviewMedia(BaseModel):
    kind: Literal["audio_segment"]
    audio_file_id: UUID
    segment_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    name: str

class ReviewPayload(BaseModel):
    metrics: tuple[ReviewMetric, ...] = Field(max_length=64)
    warnings: tuple[str, ...] = Field(max_length=64)
    groups: tuple[ReviewGroup, ...] = Field(max_length=32)
```

Persist one immutable JSONB payload and optional validated continuation graph. Lock the row for decisions and assign `review_<uuid.hex>` before commit. Add `review_count` to the paginated jobs projection with one grouped subquery, not N+1 reads.

- [ ] **Step 4: Add and apply the migration**

Create `workflow_reviews`; do not yet alter speaker audits. Import the model in `shared.db.models` so metadata discovery includes it.

Run: `nix develop --command alembic upgrade head && nix develop --command alembic check`

Expected: upgrade succeeds and Alembic reports no new operations.

- [ ] **Step 5: Verify GREEN, remove the temporary test, and commit**

Run the test command again, then `nix develop --command uv run --with ruff ruff check src/shared/db/reviews src/shared/db/jobs migrations/versions/20260714_05_add_workflow_reviews.py`.

Commit: `feat: persist generic workflow reviews`

---

### Task 2: Generic backend review API and continuation decisions

**Files:**
- Create: `src/backend/reviews/api.py`
- Create: `src/backend/reviews/schemas.py`
- Create: `src/backend/reviews/service.py`
- Create: `src/backend/reviews/__init__.py`
- Modify: `src/backend/api.py`
- Test temporarily: `tmp_tests/test_review_api.py`

**Interfaces:**
- Produces `review_router(manager: BackendManager) -> APIRouter`.
- Endpoints: `GET /reviews?run_id=`, `GET /reviews/{id}`, and `POST /reviews/{id}/decision`.

- [ ] **Step 1: Write failing API tests**

Assert list/detail response validation, 404, rejection without dispatch, approval dispatch through a fake manager, repeated approval returning the identical run, and invalid conflicting decisions returning 409.

```python
response = client.post(f"/reviews/{review_id}/decision", json={"decision": "approved"})
assert response.status_code == 202
assert response.json()["continuation_run_id"] == f"review_{review_id.hex}"
assert manager.requests[0].run_id == f"review_{review_id.hex}"
```

- [ ] **Step 2: Verify RED**

Run: `nix develop --command uv run --with pytest python -m pytest -q tmp_tests/test_review_api.py`

Expected: missing `backend.reviews`.

- [ ] **Step 3: Implement the router and decision service**

The service validates the stored `ReviewContinuation`, records the decision through CRUD, and calls `manager.start_inline_graph`. Treat `DuplicateRunError` as an idempotent retry by returning `manager.status(run_id)`. Convert missing reviews to 404 and decision conflicts to 409.

- [ ] **Step 4: Verify GREEN, remove the temporary test, and commit**

Run API tests, Ruff, and `python -m compileall -q src/backend/reviews`; then remove only `tmp_tests/test_review_api.py`.

Commit: `feat: expose generic workflow reviews`

---

### Task 3: Publish speaker audits as reviews and remove report artifacts

**Files:**
- Modify: `src/shared/db/speakers/models.py`
- Modify: `src/shared/db/speakers/schemas.py`
- Modify: `src/shared/db/speakers/audit_crud.py`
- Modify: `src/runner/nodes/models.py`
- Modify: `src/runner/nodes/speaker_clustering/audit_report/models.py`
- Modify: `src/runner/nodes/speaker_clustering/audit_report/builder.py`
- Delete: `src/runner/nodes/speaker_clustering/audit_report/render.py`
- Modify: `src/runner/nodes/speaker_clustering/cluster_runtime/audit_node.py`
- Modify: `src/runner/nodes/speaker_clustering/cluster_runtime/apply.py`
- Create: `src/runner/nodes/speaker_clustering/cluster_runtime/audit_source.py`
- Modify: `src/runner/nodes/speaker_clustering/__init__.py`
- Modify: `src/runner/nodes/registry.py`
- Modify: `migrations/versions/20260714_05_add_workflow_reviews.py`
- Test temporarily: `tmp_tests/test_speaker_review.py`

**Interfaces:**
- `SpeakerAuditRef(audit_id, cluster_run_id, review_id)` replaces artifact IDs.
- `SpeakerAuditSource` accepts `audit_id` and emits only approved, completed audits.

- [ ] **Step 1: Write failing review-publication tests**

Test typed metric/group mapping, bulk resolution of selected segment bounds, retry identity, no extra-file creation, source rejection before approval, and apply rejection when the linked review is not approved.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command uv run --with pytest python -m pytest -q tmp_tests/test_speaker_review.py`

Expected: current audit still requires report/listening artifacts.

- [ ] **Step 3: Replace file rendering with typed review construction**

Make the builder return `AssignmentAuditDocument` directly. Add a focused adapter that maps metrics and selected entries into generic review schemas and resolves all audio segments using one bulk audio CRUD call per bounded selection set.

Create a continuation request containing exactly:

```python
nodes = [
    GraphNodeRequest(id="audit", type="SpeakerAuditSource", params={"audit_id": str(audit.id)}),
    GraphNodeRequest(id="apply", type="ApplySpeakerClusters", params={"approved_audit_id": str(audit.id)}),
]
```

Persist the review with `producer_run_id=context.run_id`; complete the speaker audit with `review_id`. Remove all report/listening upload and ZIP paths.

- [ ] **Step 4: Harden source and apply approval invariants**

`SpeakerAuditSource` and apply both load the audit and linked review through CRUD, require completed/approved states, and compare IDs. Keep existing bounded apply checkpointing unchanged.

- [ ] **Step 5: Finish the migration**

Drop speaker audit report/listening foreign keys and columns, add nullable `review_id` for open audits, and require it in the completed-state check constraint. No legacy data conversion.

- [ ] **Step 6: Verify GREEN, remove the temporary test, and commit**

Run temporary tests, Ruff, compileall, Alembic upgrade/check, and schema export registration. Remove only `tmp_tests/test_speaker_review.py`.

Commit: `refactor: publish speaker audits as workflow reviews`

---

### Task 4: Generic Jobs review drawer and bounded audio playback

**Files:**
- Create: `src/frontend/src/features/jobs/ReviewDrawer.tsx`
- Create: `src/frontend/src/features/jobs/reviews.ts`
- Modify: `src/frontend/src/features/jobs/api.ts`
- Modify: `src/frontend/src/features/jobs/query.ts`
- Modify: `src/frontend/src/features/jobs/JobsScreen.tsx`
- Modify: `src/frontend/src/shared/media/WaveformPlayer.tsx`
- Test temporarily: `src/frontend/src/features/jobs/review.tmp.test.tsx`

**Interfaces:**
- Produces generic TypeScript review/media discriminated unions matching backend schemas.
- Adds `clipStart` and `clipEnd` optional props to `WaveformPlayer`.

- [ ] **Step 1: Write failing frontend tests**

Cover metric/group rendering, virtualized item rendering, audio media URL and clip bounds, decision confirmation, job/review invalidation, and approved continuation state.

- [ ] **Step 2: Verify RED**

Run the existing frontend test command through `nix develop`; expect missing `ReviewDrawer` and clip-bound behavior.

- [ ] **Step 3: Implement API/query seams and the drawer**

Keep all HTTP types and calls in `api.ts`, query/mutation ownership in `query.ts`, generic display helpers in `reviews.ts`, and rendering in `ReviewDrawer.tsx`. Use `VirtualTable` for every group list and existing cards, badges, modal, confirm, toast, and waveform components.

- [ ] **Step 4: Add clip-aware playback**

On play, seek to `clipStart`; report position relative to the clip; pause/reset at `clipEnd`. Preserve existing behavior when bounds are absent and continue using `/audio-files/{id}/content`.

- [ ] **Step 5: Integrate with Jobs and verify GREEN**

Add `review_count` to `Job`, show Review only when nonzero, and open the drawer without navigation. Run targeted tests, TypeScript checking, and the production build through Nix. Remove the temporary test.

Commit: `feat: review workflow outputs from jobs`

---

### Task 5: End-to-end review, approval, and cleanup verification

**Files:**
- Modify if needed: `workflows/speaker_cluster_ecapa.json`
- Delete obsolete code confirmed by `rg`: report renderers, report/listening artifact types, and stale documentation references.

- [ ] **Step 1: Run a labeled audit graph through `POST /graphs/runs`**

Use the existing sealed 28-segment fixture or recreate it temporarily. Verify the job succeeds and `GET /reviews?run_id=` returns one pending review with perfect labeled precision/purity and four populated review groups.

- [ ] **Step 2: Verify rejection and approval behavior**

Reject one disposable review and confirm no continuation job. Approve the target review and confirm exactly one `SpeakerAuditSource -> ApplySpeakerClusters` job is created; repeat approval and confirm the same run ID is returned.

- [ ] **Step 3: Verify assignment safety**

Inspect through public audio CRUD: 24 recurring-speaker segments share exactly three distinct voices, four unrelated singletons remain unassigned, and segment text/metadata are preserved.

- [ ] **Step 4: Run final checks**

Run Ruff, Python compileall, frontend type/build checks, JSON parsing, schema registration, Alembic upgrade/check, file/folder size checks, and `git diff --check`. Inspect run logs with `python -m cli logs <run_id>` and failures with `python -m cli failed <run_id>`.

- [ ] **Step 5: Remove task-owned temporary files and commit**

Confirm no audit HTML/JSON/ZIP/listening artifacts are generated and the worktree contains no task-owned fixtures or tests.

Commit: `chore: verify generic workflow review flow`
