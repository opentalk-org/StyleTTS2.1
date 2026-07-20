# Canonical Speaker ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicate structured speaker-name fields with the single `speaker_id` contract across the project.

**Architecture:** Preserve the distinction between source/detected `speaker_id` strings and persisted `voice_id` UUIDs. Rename the contract at every internal boundary in one breaking greenfield change, allowing only third-party adapters to consume an external `speaker` key before mapping it to `speaker_id`.

**Tech Stack:** Python 3, dataclasses, Pydantic, FastAPI, React, TypeScript, Zustand, Nix.

## Global Constraints

- Run all Python and frontend commands through `nix develop --command ...`.
- Do not add compatibility aliases or fallback reads for `speaker`, `speakerName`, or `speakerId`.
- Keep `voice_id` unchanged because it identifies a persisted voice record.
- Do not keep temporary tests in the repository.
- Preserve human-facing uses of the noun “speaker” and external-library boundary fields.

---

### Task 1: Runtime and persistence contract

**Files:**
- Modify: `src/runner/nodes/models.py`
- Modify: `src/shared/db/audio/schemas.py`
- Modify: `src/shared/db/audio/crud.py`
- Modify: `src/runner/nodes/audio_segments/writeback_helpers.py`
- Modify: `src/runner/nodes/audio_segments/extract_writeback.py`
- Modify: `src/runner/nodes/audio_segments/external_record.py`
- Modify: `src/runner/nodes/audio_segments/extract.py`
- Test: temporary `tests/test_speaker_id_contract.py`

**Interfaces:**
- Consumes: persisted audio metadata and segment dictionaries.
- Produces: `AudioSegment.speaker_id: str | None`, `Transcript.speaker_id: str | None`, and Pydantic segment payloads with `speaker_id: str`.

- [ ] **Step 1: Write the failing contract test**

```python
from runner.nodes.models import AudioSegment, Transcript
from shared.db.audio.schemas import SegmentCreate


def test_runtime_and_schema_expose_only_speaker_id():
    assert "speaker_id" in AudioSegment.__dataclass_fields__
    assert "speaker" not in AudioSegment.__dataclass_fields__
    assert "speaker_id" in Transcript.__dataclass_fields__
    assert "speaker" not in Transcript.__dataclass_fields__
    assert "speaker_id" in SegmentCreate.model_fields
    assert "speaker" not in SegmentCreate.model_fields
```

- [ ] **Step 2: Run the test and confirm the old contract fails**

Run: `nix develop --command python tests/test_speaker_id_contract.py`

Expected: FAIL because the models still expose `speaker`.

- [ ] **Step 3: Rename the runtime and persistence fields**

Replace `speaker` with `speaker_id` in the dataclass/Pydantic declarations and in all listed segment dictionary serialization/deserialization sites. Change the speaker sort key in CRUD from `speaker` to `speaker_id`.

- [ ] **Step 4: Run the focused test**

Run: `nix develop --command python tests/test_speaker_id_contract.py`

Expected: PASS.

### Task 2: Runner producers and consumers

**Files:**
- Modify: `src/runner/nodes/asr/transcript.py`
- Modify: `src/runner/nodes/audio_processing/nodes.py`
- Modify: `src/runner/nodes/audio_segments/speaker_split.py`
- Modify: `src/runner/nodes/dataset_writeback/nodes.py`
- Modify: `src/runner/nodes/hetzner/ds_v1_segments.py`
- Modify: `src/runner/nodes/hetzner/ds_v2_audio.py`
- Modify: `src/runner/nodes/youtube_source/nodes.py`
- Modify: `src/runner/nodes/speaker_clustering/embed_node.py`
- Modify: `src/runner/nodes/statistics/aggregate.py`
- Modify: `src/runner/nodes/statistics/aggregate_helpers.py`
- Modify: `src/runner/nodes/statistics/segments.py`
- Modify: `src/runner/nodes/statistics/voice_embedding_plot.py`
- Modify: `src/runner/nodes/testing/nodes.py`
- Modify: `src/runner/nodes/training/common/manifest/build.py`
- Modify: `workflows/voice_embedding_pca.json`

**Interfaces:**
- Consumes: the Task 1 `speaker_id` runtime/storage contract.
- Produces: nodes and manifests that read/write only `speaker_id`; external Sortformer results are mapped at the adapter boundary.

- [ ] **Step 1: Extend the temporary test with source-level assertions**

```python
from pathlib import Path


def test_runner_does_not_serialize_legacy_speaker_key():
    roots = [Path("src/runner/nodes"), Path("workflows")]
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".json"}:
                continue
            text = path.read_text()
            if '"speaker":' in text and "Sortformer" not in text:
                offenders.append(str(path))
    assert offenders == []
```

- [ ] **Step 2: Run the test and confirm producers still fail**

Run: `nix develop --command python tests/test_speaker_id_contract.py`

Expected: FAIL with runner files containing serialized `speaker` keys.

- [ ] **Step 3: Migrate all runner data paths**

Rename fields, locals, settings literals, and metadata keys that carry speaker identity to `speaker_id`. Keep prose and the Sortformer response adapter's external `item["speaker"]` read. Remove duplicated `speaker` writes where `speaker_id` already exists.

- [ ] **Step 4: Run the focused test**

Run: `nix develop --command python tests/test_speaker_id_contract.py`

Expected: PASS.

### Task 3: Backend and frontend contract

**Files:**
- Modify: `src/backend/audio/schemas.py`
- Modify: `src/backend/audio/api.py`
- Modify: `src/backend/mos/schemas.py`
- Modify: `src/backend/mos/api.py`
- Modify: `src/frontend/src/features/audio/api.ts`
- Modify: `src/frontend/src/features/audio/store.ts`
- Modify: `src/frontend/src/features/audio/logic.ts`
- Modify: `src/frontend/src/features/audio/actions.ts`
- Modify: `src/frontend/src/features/audio/AudioToolbar.tsx`
- Modify: `src/frontend/src/features/audio/AudioRow.tsx`
- Modify: `src/frontend/src/features/audio/SegmentEditor.tsx`
- Modify: `src/frontend/src/features/audio/SegmentRow.tsx`
- Modify: `src/frontend/src/features/audio/SegmentTimeline.tsx`
- Modify: `src/frontend/src/features/audio/editorStore.ts`
- Modify: `src/frontend/src/features/audio/editor/EditorHeader.tsx`
- Modify: `src/frontend/src/features/mos/api.ts`
- Modify: `src/frontend/src/features/mos/MosAudioCard.tsx`
- Modify: `src/frontend/src/features/workflows/components/WorkflowFieldPickers.tsx`

**Interfaces:**
- Consumes: canonical stored `speaker_id` metadata and segments.
- Produces: API response/request and UI state properties named `speaker_id`, including upload form and sort value.

- [ ] **Step 1: Extend the temporary contract test for backend fields**

```python
from backend.audio.schemas import AudioFileListItem, AudioSegmentRead, AudioSort
from backend.mos.schemas import MosAudioRead


def test_backend_exposes_only_speaker_id():
    assert "speaker_id" in AudioFileListItem.model_fields
    assert "speaker" not in AudioFileListItem.model_fields
    assert "speaker_id" in AudioSegmentRead.model_fields
    assert "speaker_id" in MosAudioRead.model_fields
    assert "speaker_id" in AudioSort.__args__
    assert "speaker" not in AudioSort.__args__
```

- [ ] **Step 2: Run the test and confirm the API contract fails**

Run: `nix develop --command python tests/test_speaker_id_contract.py`

Expected: FAIL because backend responses still expose `speaker`.

- [ ] **Step 3: Rename backend and frontend contract fields**

Update schemas, serializers, form names, TypeScript types, state updates, sort values, components, and labels whose structured key is `speaker`. Keep displayed copy such as “Speaker” where it is a human-facing noun.

- [ ] **Step 4: Run focused and static verification**

Run: `nix develop --command python -m pytest tests/test_speaker_id_contract.py -q`

Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`

Expected: PASS for the contract test and frontend build.

### Task 4: Full verification and cleanup

**Files:**
- Delete: temporary `tests/test_speaker_id_contract.py`

**Interfaces:**
- Consumes: all migrated project layers.
- Produces: a clean worktree change with no committed temporary tests.

- [ ] **Step 1: Search for obsolete structured keys**

Run: `rg -n --glob '*.py' --glob '*.ts' --glob '*.tsx' --glob '*.json' '("speaker"\s*:|[.]speaker\b|\bspeaker\s*:)' src workflows`

Expected: only external-library boundaries, human-facing schema descriptions, or algorithm-local domain terminology; no internal data contract fields.

- [ ] **Step 2: Run the relevant Python suite**

Run: `nix develop --command python -m compileall -q src`

Expected: PASS.

- [ ] **Step 3: Remove the temporary test and re-run verification**

Delete `tests/test_speaker_id_contract.py`, then run `nix develop --command python -m compileall -q src` and `nix develop --command bash -lc 'cd src/frontend && npm run build'`.

Expected: both commands PASS.

- [ ] **Step 4: Review the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; the pre-existing `IDEA.md` edit remains untouched.
