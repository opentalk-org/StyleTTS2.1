# MOS Workflow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect MOS comparison data and the Wav2Vec2 regressor to registered training/inference nodes, the Training UI, and real graph verification.

**Architecture:** Consume the shared MOS CRUD, `mos_base` checkpoint, and MOS model utilities delivered by Tasks 1-4 of the foundation plan. Reuse generic runflow ports and lifecycle policies; keep database writeback behind shared CRUD.

**Tech Stack:** PyTorch, Transformers Wav2Vec2, runflow, React/TypeScript, TanStack Query, FastAPI/PostgreSQL.

## Global Constraints

- Complete `2026-07-10-mos-rating-training-implementation.md` Tasks 1-4 first.
- Run all Python, pytest, npm, and node commands through `nix develop --command ...`.
- Keep files below 300 lines and folders below 16 files; keep MOS out of `src/runflow`.
- Reuse generic ports and shared CRUD/storage facades; preserve unrelated worktree changes.
- Keep temporary tests under `.tmp_mos_tests/` and remove them before completion.
- Test nodes through registered graphs, never by calling `execute()` directly.

---

### Task 5: MOS manifest and training node

**Files:**
- Create: `src/runner/nodes/mos/{manifest,dataset,train,nodes}.py`
- Modify: `src/runner/nodes/mos/__init__.py`, `src/runner/nodes/registry.py`
- Test: `.tmp_mos_tests/test_mos_training.py`

**Interfaces:** `BuildMosTrainingManifest` emits `training_manifest`; `MosModelTraining` consumes `checkpoint` and `training_manifest`, emits `training_result` with checkpoint type `mos_model`.

- [ ] **Step 1: Write failing generic-port schema tests**

```python
def test_mos_nodes_use_generic_ports():
    assert BuildMosTrainingManifestNode.OUTPUTS["training_manifest"].TYPE_NAME == "TRAINING_MANIFEST"
    assert MosModelTrainingNode.INPUTS["checkpoint"].TYPE_NAME == "CHECKPOINT_REF"
```

Run: `nix develop --command python .tmp_mos_tests/test_mos_training.py`
Expected: FAIL because nodes do not exist.

- [ ] **Step 2: Implement deterministic JSONL manifests and lazy paired batches**

```json
{"comparison_id":"...","audio_a_id":"...","audio_b_id":"...","score_a":4.0,"score_b":2.5,"preferred":"a"}
```

Require two comparisons; reserve `min(validation_comparisons, count - 1)`; store paths/counts in `TrainingManifest.metadata`; bulk-read pair audio with `audio_crud.bulk_read_audio_files` in the collator.

- [ ] **Step 3: Implement cancellable training and checkpoint publication**

Train encoder/head in batches, validate each epoch, report progress, save interval/final states, and call `publish_training_result(..., "mos_model", ...)` with accelerator cleanup in `finally`.

- [ ] **Step 4: Register nodes and verify GREEN/schema export**

Run: `nix develop --command python .tmp_mos_tests/test_mos_training.py`
Run: `nix develop --command python -c 'from runner.nodes.registry import create_node_registry; r=create_node_registry(); print(r.nodes["MosModelTraining"])'`
Expected: PASS and registered node output.

- [ ] **Step 5: Commit Task 5 paths**

```bash
git commit --only src/runner/nodes/mos src/runner/nodes/registry.py -m "feat: train MOS models through workflows"
```

### Task 6: Score-overwriting inference node

**Files:**
- Create: `src/runner/nodes/mos/inference.py`
- Modify: `src/runner/nodes/mos/__init__.py`, `src/runner/nodes/registry.py`, `src/shared/db/audio/crud.py`
- Test: `.tmp_mos_tests/test_mos_inference.py`

**Interfaces:** `PredictMosScore` inputs `audio` plus broadcast `checkpoint`; outputs one `audio` and `writeback_result` per input; `bulk_update_audio_scores(Session, dict[UUID, float])`.

- [ ] **Step 1: Write failing schema and bulk-write contract tests**

```python
def test_predict_node_preserves_audio_shape():
    assert set(PredictMosScoreNode.OUTPUTS) == {"audio", "writeback_result"}
    assert PredictMosScoreNode.BATCH_POLICY.max_size > 1
```

Run: `nix develop --command python .tmp_mos_tests/test_mos_inference.py`
Expected: FAIL because the node is missing.

- [ ] **Step 2: Implement lifecycle loading, batched prediction, and bulk score writeback**

Validate type `mos_model`, load once per checkpoint ID, require loaded bytes, chunk inference, check cancellation and report progress, bulk update scores once, and preserve input order.

- [ ] **Step 3: Verify GREEN and commit**

Run: `nix develop --command python .tmp_mos_tests/test_mos_inference.py`
Expected: PASS.

```bash
git commit --only src/runner/nodes/mos src/runner/nodes/registry.py src/shared/db/audio/crud.py -m "feat: infer and overwrite MOS scores"
```

### Task 7: MOS Training tab workflow form

**Files:**
- Create: `src/frontend/src/features/training/MosModelForm.tsx`
- Modify/split: `src/frontend/src/features/training/TrainingScreen.tsx`, `src/frontend/src/features/training/store.ts`, `src/frontend/src/features/training/logic.ts`, `src/frontend/src/features/training/QueueCard.tsx`
- Test: `src/frontend/src/training-mos-contract.ts` (temporary)

**Interfaces:** tab value `mos`; graph nodes `TrainingRunInput`, `SelectTrainingDataset`, `SelectCheckpoint`, `BuildMosTrainingManifest`, `MosModelTraining`.

- [ ] **Step 1: Add a failing graph contract and verify RED**

```ts
import { TRAINING_WORKFLOWS } from "@/features/training/logic";
if (TRAINING_WORKFLOWS.mos.nodes.at(-1)?.type !== "MosModelTraining") throw new Error("missing MOS graph");
```

Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: FAIL because `mos` is not a `TrainTab`.

- [ ] **Step 2: Implement graph spec, form, and MOS-aware queue validation**

Reuse `datasetOptions`, `checkpointOptions(..., "mos_base", ...)`, `SettingField`, `FormSection`, and `QueueCard`. Require display name, dataset, and base checkpoint; do not require alphabet or OOD assets. Split `logic.ts` before it crosses 300 lines.

- [ ] **Step 3: Remove contract, verify GREEN, and commit**

Run: `rm src/frontend/src/training-mos-contract.ts`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: exit 0.

```bash
git commit --only src/frontend/src/features/training -m "feat: add MOS training workflow tab"
```

### Task 8: End-to-end verification and cleanup

**Files:**
- Create temporarily: `workflows/mos_inference_smoke.json`
- Remove: `.tmp_mos_tests/`, temporary workflow, frontend contract files
- Inspect: all paths changed by Tasks 1-7

- [ ] **Step 1: Run focused and static verification**

Run: `nix develop --command python -m unittest discover -s .tmp_mos_tests -p 'test_*.py'`
Run: `nix develop --command python -m compileall -q src`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: all exit 0.

- [ ] **Step 2: Run the shared stack and migration**

Run: `nix develop --command runflow-dev-status`
Run if absent: `nix develop --command runflow-dev-session`
Run: `nix develop --command alembic current`
Expected: healthy stack and revision `91e06b9c7440`.

- [ ] **Step 3: Exercise annotation and inference through public APIs/graph submission**

Create two throwaway audios in one dataset through public APIs, request `/mos/pair`, submit `/mos/ratings`, verify both scores and one comparison, then submit `AudioSource -> LoadAudio -> PredictMosScore` through `POST /graphs/runs` with a trained `mos_model` checkpoint.

Run: `nix develop --command python -m cli runs`
Run: `nix develop --command python -m cli logs <run_id>`
Run on failure: `nix develop --command python -m cli failed <run_id>`
Expected: terminal success and overwritten scalar scores.

- [ ] **Step 4: Exercise one-epoch real training**

Download the base with `CatalogDownload`, create at least two ratings, queue the MOS training graph for one epoch, inspect its `mos_model` checkpoint, and use it in Step 3. This proves catalog, local loading, combined loss, publication, and inference together.

- [ ] **Step 5: Remove temporary assets and audit requirements**

Run: `rm -rf .tmp_mos_tests workflows/mos_inference_smoke.json`
Run: `rg -n "MOS|Mos|mos_models|wav2vec2-xls-r-300m|PredictMosScore" src migrations workflows`
Run: `git diff --check d78d221..HEAD`
Expected: no temporary test/workflow files, no whitespace errors, and evidence for every approved spec requirement.
