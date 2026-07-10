# MOS Rating, Training, and Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build dataset-scoped pair rating, Wav2Vec2 MOS training, catalog integration, and score-overwriting workflow inference.

**Architecture:** Store explicit pair comparisons beside the existing scalar audio score, expose them through a focused backend/frontend feature, and reuse generic workflow ports for a MOS manifest, trainer, and inference node. The runner keeps MOS-specific code under `runner/nodes/mos/`; `runflow` remains unchanged.

**Tech Stack:** FastAPI, SQLAlchemy/Alembic/PostgreSQL, Pydantic, React/TypeScript, TanStack Query, Zustand, PyTorch, Transformers Wav2Vec2, runflow.

## Global Constraints

- Run every Python, pytest, npm, and node command through `nix develop --command ...`.
- Work in the current checkout; do not create a branch or worktree.
- Keep files below 300 lines and folders below 16 files.
- Use shared feature CRUD facades for PostgreSQL, audio bytes, scores, checkpoints, and storage.
- Keep MOS concepts out of `src/runflow`; reuse `AudioPort`, `CheckpointRefPort`, `TrainingManifestPort`, and `TrainingResultPort`.
- Develop behavior test-first with `.tmp_mos_tests/`, then remove that directory before completion.
- Test nodes through registered graphs, never by calling `execute()` directly.
- Preserve unrelated staged and unstaged user changes; commits use explicit pathspecs only.

---

### Task 1: MOS persistence and backend API

**Files:**
- Create: `src/shared/db/mos/{__init__,models,schemas,crud}.py`
- Create: `src/backend/mos/{__init__,schemas,api}.py`
- Create: `migrations/versions/20260710_1200_91e06b9c7440_add_mos_comparisons.py`
- Modify: `src/shared/db/connection.py`, `src/backend/api.py`
- Test: `.tmp_mos_tests/test_mos_persistence.py`

**Interfaces:** `sample_pair(Session, list[UUID]) -> MosPair`; `create_rating(Session, MosRatingCreate) -> MosComparison`; pair/create plus paginated list, latest update, and latest undo endpoints under `/mos`.

- [ ] **Step 1: Write the failing persistence contract**

```python
from uuid import uuid4

from shared.db.mos.schemas import MosRatingCreate

def test_rating_accepts_distinct_pair_and_member_preference():
    ids = [uuid4() for _ in range(3)]
    value = MosRatingCreate(dataset_id=ids[0], audio_a_id=ids[1], audio_b_id=ids[2], preferred_audio_id=ids[1], score_a=3.5, score_b=2.0)
    assert value.preferred_audio_id == value.audio_a_id
```

- [ ] **Step 2: Verify RED**

Run: `nix develop --command python .tmp_mos_tests/test_mos_persistence.py`
Expected: FAIL because `shared.db.mos` does not exist.

- [ ] **Step 3: Implement model, CRUD, migration, and router**

```python
class MosComparison(Base):
    __tablename__ = "mos_comparisons"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    audio_a_id: Mapped[UUID] = mapped_column(ForeignKey("audio_files.id", ondelete="CASCADE"))
    audio_b_id: Mapped[UUID] = mapped_column(ForeignKey("audio_files.id", ondelete="CASCADE"))
    preferred_audio_id: Mapped[UUID] = mapped_column(ForeignKey("audio_files.id", ondelete="CASCADE"))
    score_a: Mapped[float] = mapped_column(Float)
    score_b: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
```

Use Pydantic validators for finite scores/distinct IDs, SQL check constraints for pair invariants, indexed UUID-threshold sampling, membership checks against `dataset_audio_files`, and one CRUD-owned commit that updates both `AudioFile.score` values and inserts the comparison. Persist both previous scores so the newest comparison can be undone deterministically.

- [ ] **Step 4: Verify GREEN and migration head**

Run: `nix develop --command python .tmp_mos_tests/test_mos_persistence.py`
Run: `nix develop --command alembic heads`
Expected: PASS; one head at `91e06b9c7440`.

- [ ] **Step 5: Commit only Task 1 paths**

```bash
git commit --only src/shared/db/mos src/backend/mos src/shared/db/connection.py src/backend/api.py migrations/versions/20260710_1200_91e06b9c7440_add_mos_comparisons.py -m "feat: persist MOS pair ratings"
```

### Task 2: MOS annotation screen and shared score input

**Files:**
- Create: `src/frontend/src/features/mos/{api,query,store,logic,MosAudioCard,MosScreen}.ts{x,}`
- Create: `src/frontend/src/features/audio/AudioScoreInput.tsx`
- Modify: `src/frontend/src/features/audio/SegmentEditor.tsx`
- Modify: `src/frontend/src/app/nav.ts`, `src/frontend/src/app/navStore.ts`, `src/frontend/src/app/ScreenRouter.tsx`
- Test: `src/frontend/src/mos-contract.ts` (temporary)

**Interfaces:** `fetchMosPair(datasetIds: string[]): Promise<MosPair>`; `saveMosRating(MosRatingRequest): Promise<MosRating>`; reusable `AudioScoreInput` accepts draft text, disabled state, and change/commit/cancel callbacks.

- [ ] **Step 1: Add a failing TypeScript contract importing the missing MOS API and screen**

```ts
import { fetchMosPair, saveMosRating } from "@/features/mos/api";
import { MosScreen } from "@/features/mos/MosScreen";
void [fetchMosPair, saveMosRating, MosScreen];
```

- [ ] **Step 2: Verify RED**

Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: FAIL with unresolved `@/features/mos/*` imports.

- [ ] **Step 3: Implement the feature**

```ts
export type MosRatingRequest = {
  dataset_id: string; audio_a_id: string; audio_b_id: string;
  preferred_audio_id: string; score_a: number; score_b: number;
};
```

Use dataset checkboxes, `WaveformPlayer`, `backendResourceUrl`, the extracted score control, preferred A/B buttons, TanStack Query mutations, audio-query invalidation, and a Zustand-only selection/draft store. Add `mos` to navigation and routing.

- [ ] **Step 4: Remove the contract and verify GREEN**

Run: `rm src/frontend/src/mos-contract.ts`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: TypeScript and Vite exit 0.

- [ ] **Step 5: Commit only annotation UI paths**

```bash
git commit --only src/frontend/src/features/mos src/frontend/src/features/audio/AudioScoreInput.tsx src/frontend/src/features/audio/SegmentEditor.tsx src/frontend/src/app/nav.ts src/frontend/src/app/navStore.ts src/frontend/src/app/ScreenRouter.tsx -m "feat: add MOS comparison screen"
```

### Task 2b: Comparison history, change/undo, and combined choice submission

**Files:**
- Modify: `src/shared/db/mos/{models,schemas,crud}.py`, `src/backend/mos/{schemas,api}.py`, `migrations/versions/20260710_1200_91e06b9c7440_add_mos_comparisons.py`
- Create: `src/frontend/src/features/mos/{MosHistoryList,MosHistoryRow}.tsx`
- Modify: `src/frontend/src/features/mos/{api,query,store,logic,MosAudioCard,MosScreen}.ts{x,}`
- Test: `.tmp_mos_tests/test_mos_history.py`, `src/frontend/src/mos-history-contract.ts` (temporary)

**Interfaces:** `list_comparisons_page`; `update_latest_rating`; `undo_latest_rating`; `GET/PATCH/DELETE /mos/ratings`; infinite history query.

- [ ] **Step 1: Write failing latest-mutation and frontend contracts**

```python
def test_history_schema_preserves_previous_scores():
    assert "previous_score_a" in MosComparison.__table__.columns
    assert "previous_score_b" in MosComparison.__table__.columns
```

```ts
import { fetchMosRatings, updateMosRating, undoMosRating } from "@/features/mos/api";
void [fetchMosRatings, updateMosRating, undoMosRating];
```

- [ ] **Step 2: Verify RED, then implement paginated history and latest-only mutations**

Run: `nix develop --command python .tmp_mos_tests/test_mos_history.py`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: missing columns/functions. Update stores previous scores on create; PATCH rewrites newest labels/scores; DELETE restores previous scores and removes newest row.

- [ ] **Step 3: Implement virtual history and choose-and-save card actions**

Use `useInfiniteQuery` plus `useVirtualizer`. Older rows are read-only; the newest row owns inline score/preference editing and undo. Replace the separate preference/save flow with one submit action per audio card.

- [ ] **Step 4: Remove frontend contract and verify GREEN**

Run: `nix develop --command python .tmp_mos_tests/test_mos_history.py`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: both exit 0; all feature files stay below 300 lines.

- [ ] **Step 5: Commit Task 2b paths**

```bash
git commit --only src/shared/db/mos src/backend/mos migrations/versions/20260710_1200_91e06b9c7440_add_mos_comparisons.py src/frontend/src/features/mos -m "feat: manage MOS comparison history"
```

### Task 3: Wav2Vec2 MOS base catalog

**Files:**
- Modify: `src/runner/nodes/assets/catalog.py`, `src/runner/nodes/assets/catalog_runtime/tasks.py`
- Modify: `src/frontend/src/features/checkpoints/logic.ts`, `src/frontend/src/features/checkpoints/CheckpointsScreen.tsx`
- Test: `.tmp_mos_tests/test_mos_catalog.py`

**Interfaces:** catalog key `mos_models`; item `facebook/wav2vec2-xls-r-300m`; checkpoint type `mos_base`.

- [ ] **Step 1: Write a failing catalog/schema test**

```python
def test_mos_catalog_is_registered():
    assert "mos_models" in CATALOG_DOWNLOAD_TASKS
    assert CatalogKey.MOS_MODELS.value == "mos_models"
```

- [ ] **Step 2: Verify RED, implement download task, then verify GREEN**

Run: `nix develop --command python .tmp_mos_tests/test_mos_catalog.py`
Expected RED: missing key. Implement `ensure_model_checkpoint("mos_base", item, lambda p: download_hf_snapshot(item, p, ignore_patterns=["*.msgpack", "*.h5"]))`; reject any model ID except the specified Facebook repository. Re-run and expect PASS.

- [ ] **Step 3: Add the catalog card and checkpoint filters**

```ts
{ name: "Wav2Vec2 XLS-R 300M · MOS base", file: "facebook/wav2vec2-xls-r-300m", group: "Training assets", catalogKey: "mos_models", item: "facebook/wav2vec2-xls-r-300m" }
```

Add `mos_base` and `mos_model` tones and filter choices.

- [ ] **Step 4: Verify backend schema and frontend build**

Run: `nix develop --command python -c 'from runner.nodes.registry import create_node_registry; print(len(create_node_registry().nodes))'`
Run: `nix develop --command bash -lc 'cd src/frontend && npm run build'`
Expected: both exit 0.

- [ ] **Step 5: Commit Task 3 paths**

```bash
git commit --only src/runner/nodes/assets/catalog.py src/runner/nodes/assets/catalog_runtime/tasks.py src/frontend/src/features/checkpoints/logic.ts src/frontend/src/features/checkpoints/CheckpointsScreen.tsx -m "feat: catalog Wav2Vec2 MOS base"
```

### Task 4: MOS model, audio preparation, and pair loss

**Files:**
- Create: `src/runner/nodes/mos/{__init__,audio,model,loss}.py`
- Test: `.tmp_mos_tests/test_mos_model.py`

**Interfaces:** `MosRegressor.forward(input_values, attention_mask) -> Tensor`; `mos_pair_loss(pred_a, pred_b, score_a, score_b, preferred_sign, comparison_weight) -> MosLoss`.

- [ ] **Step 1: Write and run a failing pure loss test**

```python
def test_pair_loss_rewards_the_preferred_order():
    correct = mos_pair_loss(tensor([4.]), tensor([2.]), tensor([4.]), tensor([2.]), tensor([1.]), 1.).total
    reversed_ = mos_pair_loss(tensor([2.]), tensor([4.]), tensor([4.]), tensor([2.]), tensor([1.]), 1.).total
    assert correct < reversed_
```

Run: `nix develop --command python .tmp_mos_tests/test_mos_model.py`
Expected: FAIL because the module is missing.

- [ ] **Step 2: Implement the stable loss and pooled regressor**

```python
pair = F.softplus(-preferred_sign * (pred_a - pred_b)).mean()
mos = F.mse_loss(pred_a, score_a) + F.mse_loss(pred_b, score_b)
return MosLoss(total=mos + comparison_weight * pair, mos=mos, comparison=pair)
```

Use attention-mask-aware mean pooling, a linear head, 16 kHz mono decoding/resampling, and `save_pretrained` plus `mos_head.pt`/`mos_config.json` persistence.

- [ ] **Step 3: Verify GREEN and commit**

Run: `nix develop --command python .tmp_mos_tests/test_mos_model.py`
Expected: PASS.

```bash
git commit --only src/runner/nodes/mos -m "feat: add Wav2Vec2 MOS model"
```

## Continuation

After Tasks 1-4 pass, continue with [MOS Workflow Integration Implementation Plan](./2026-07-10-mos-workflow-integration-implementation.md). It consumes the persistence, catalog, and model interfaces defined above.
