# Shared Audio Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give audio and segments one non-divergent annotation contract containing `speaker_id`, `voice_id`, `score`, `accuracy`, and `metadata`, with no `confidence` field.

**Architecture:** Define one frozen Pydantic `AudioAnnotations` value object in `shared`, compose it into `Audio`, `AudioRecordRef`, and `AudioSegment`, and expose read-only convenience properties through a shared mixin. Audio database rows store queryable annotation values in columns and custom metadata in JSONB; segment JSON and backend/frontend contracts store one nested `annotations` object.

**Tech Stack:** Python 3, Pydantic, dataclasses, SQLAlchemy, PostgreSQL JSONB, FastAPI, React, TypeScript, Nix.

## Global Constraints

- Run commands through `nix develop --command ...`.
- Do not retain `confidence` aliases or legacy storage fallbacks.
- Store quality/MOS in `score`; store transcription/alignment certainty in `accuracy`.
- Do not keep temporary tests.
- Preserve unrelated working-tree changes.

---

### Task 1: Shared annotation model and persistence

**Files:**
- Create: `src/shared/audio_annotations.py`
- Modify: `src/runner/nodes/models.py`
- Modify: `src/shared/db/audio/models.py`
- Modify: `src/shared/db/audio/schemas.py`
- Modify: `src/shared/db/audio/pack_crud.py`
- Modify: `src/shared/db/audio/external_crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Modify: `src/shared/db/audio/references_crud.py`
- Test: temporary `tests/test_score_accuracy_contract.py`

**Interfaces:**
- Produces: `AudioAnnotations(speaker_id, voice_id, score, accuracy, metadata)`, composed runtime models, and audio-row codecs.

- [ ] **Step 1: Make the temporary contract require composition**

```python
assert "annotations" in Audio.__dataclass_fields__
assert "annotations" in AudioSegment.__dataclass_fields__
assert "score" not in Audio.__dataclass_fields__
assert "confidence" not in Audio.__dataclass_fields__
```

- [ ] **Step 2: Run `nix develop --command python tests/test_score_accuracy_contract.py` and observe the flat contract fail.**

- [ ] **Step 3: Implement the shared model, compose runtime types, add audio-row columns, and make database payloads carry `annotations: AudioAnnotations`.**

- [ ] **Step 4: Run the temporary contract and confirm exit 0.**

### Task 2: API, frontend, and all node data paths

**Files:**
- Modify: `src/backend/audio/api.py`
- Modify: `src/backend/audio/schemas.py`
- Modify: `src/backend/mos/api.py`
- Modify: `src/backend/mos/schemas.py`
- Modify: `src/frontend/src/features/audio/api.ts`
- Modify: `src/frontend/src/features/audio/*.tsx`
- Modify: `src/frontend/src/features/audio/editor/*.tsx`
- Modify: `src/frontend/src/features/audio/editorStore.ts`
- Modify: `src/frontend/src/features/mos/*`
- Modify: every `src/runner/nodes/**/*.py` constructor, serializer, and consumer that handles audio annotations.

**Interfaces:**
- Consumes: Task 1 `AudioAnnotations`.
- Produces: nested API/frontend annotations and nodes that construct/update the shared object rather than independent fields or raw annotation metadata keys.

- [ ] **Step 1: Extend the temporary test to reject structured `confidence`, flat segment annotation keys, and runtime constructors without `annotations`.**

- [ ] **Step 2: Run the contract and observe the remaining producers fail.**

- [ ] **Step 3: Route sources/imports/MOS to `score`, ASR/alignment/statistical certainty to `accuracy`, and preserve the full object through extraction, writeback, clustering, manifests, and UI editing.**

- [ ] **Step 4: Run the contract, Python compilation, all 85 runner registrations, and the frontend build.**

### Task 3: Cleanup and final verification

**Files:**
- Delete: temporary `tests/test_score_accuracy_contract.py`

- [ ] **Step 1: Search `src` for obsolete structured `confidence` and flat serialized annotation keys.**

- [ ] **Step 2: Remove the temporary test with `apply_patch`.**

- [ ] **Step 3: Freshly run Python compilation, schema/registry assertions, frontend production build, and `git diff --check`.**

- [ ] **Step 4: Confirm the unrelated `IDEA.md` edit remains untouched and report any unavailable live graph verification.**
