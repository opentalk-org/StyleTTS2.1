# Speaker Embedding Shards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream dataset segments, embed them with ECAPA-TDNN in true GPU batches, and seal bounded Parquet shards behind one typed embedding-set reference.

**Architecture:** A metadata-only source keyset-pages audio records and expands segments without collecting the corpus. ECAPA processes duration-bucketed micro-batches and writes float16 Parquet shards; PostgreSQL stores idempotent run/shard state and emits a sealed reference only when durable counts match.

**Tech Stack:** Python 3.12, runflow, SpeechBrain ECAPA-TDNN, PyTorch/torchaudio, PyArrow/Parquet, SQLAlchemy/PostgreSQL, existing asset/audio CRUD

## Global Constraints

- Run Python, dependency, migration, and graph commands through `nix develop --command ...`.
- Use 16 kHz mono input, one padded `encode_batch` call per micro-batch, explicit float32 L2 normalization, and float16 shard storage.
- Do not load all segment IDs or embeddings into memory.
- All persistence uses shared CRUD and asset helpers; temporary tests are removed before handoff.
- Files stay below 300 lines and folders below 16 files.

---

### Task 1: Embedding run schema and typed graph references

**Files:**
- Create: `src/shared/db/speakers/__init__.py`
- Create: `src/shared/db/speakers/models.py`
- Create: `src/shared/db/speakers/schemas.py`
- Create: `src/shared/db/speakers/crud.py`
- Create: `migrations/versions/20260714_01_add_speaker_embedding_runs.py`
- Modify: `src/shared/db/base.py`
- Modify: `src/runner/nodes/models.py`
- Modify: `src/runner/nodes/datatypes.py`
- Test: `tmp_tests/test_speaker_embedding_storage.py`

**Interfaces:**
- Produces: `SpeakerEmbeddingShardRef(run_id, artifact_id, row_count, dimension, model_revision, preprocessing_version)` and `SpeakerEmbeddingSetRef(run_id, artifact_ids, dimension, item_count, model_revision, preprocessing_version)` plus matching ports.
- Produces: `create_embedding_run`, `register_embedding_shard`, `get_embedding_run`, `list_embedding_shards`, and `seal_embedding_run` CRUD functions.

- [ ] **Step 1: Write a failing idempotency/sealing test**

```python
def test_embedding_run_seals_only_at_expected_count(session):
    run = create_embedding_run(session, embedding_run(expected_count=3))
    register_embedding_shard(session, run.id, shard(artifact_id=A, row_count=2))
    register_embedding_shard(session, run.id, shard(artifact_id=A, row_count=2))
    with pytest.raises(ValueError, match="expected 3, stored 2"):
        seal_embedding_run(session, run.id)
```

- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_embedding_storage.py -q` and confirm missing imports**
- [ ] **Step 3: Add migration, unique `(run_id, artifact_id)` shard registration, typed schemas/CRUD, dataclasses, and ports**

```python
def seal_embedding_run(session: Session, run_id: UUID) -> SpeakerEmbeddingRun: ...
def list_embedding_shards(session: Session, run_id: UUID) -> list[SpeakerEmbeddingShard]: ...
```

- [ ] **Step 4: Run `nix develop --command alembic upgrade head && nix develop --command pytest tmp_tests/test_speaker_embedding_storage.py -q`; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: add speaker embedding run storage"`**

### Task 2: Keyset-paged segment source

**Files:**
- Create: `src/shared/db/audio/segment_references_crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Create: `src/runner/nodes/speaker_clustering/source.py`
- Create: `src/runner/nodes/speaker_clustering/__init__.py`
- Test: `tmp_tests/test_speaker_segment_source.py`

**Interfaces:**
- Produces: `SpeakerSegmentSource`, streaming `AudioPort` items containing exactly one segment with stable audio/segment IDs and `source_segment_count` metadata.

- [ ] **Step 1: Write a failing test for two DB pages and bounded audio reads**

```python
async def test_source_pages_segments_without_materializing_all(source_context):
    outputs = await drain_source(SpeakerSegmentSource, source_context)
    assert [item["audio"].segments[0].id for item in outputs] == EXPECTED_IDS
    assert max(source_context.observed_page_size) <= 1024
```

- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_segment_source.py -q`; expect missing node failure**
- [ ] **Step 3: Implement composite keyset `(audio_file_id, segment_index)`, dataset scope, count query, bounded expansion, cancellation, and progress**

```python
def list_segment_references_page(session: Session, dataset_id: UUID,
    after: SegmentCursor | None, limit: int) -> list[SegmentReference]: ...
```

- [ ] **Step 4: Rerun the source test with an empty-segment audio and verify it is skipped by query semantics; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: stream stored segments for speaker embedding"`**

### Task 3: ECAPA preprocessing and batched inference

**Files:**
- Create: `src/runner/nodes/speaker_clustering/ecapa_runtime.py`
- Create: `src/runner/nodes/speaker_clustering/embedding_rows.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tmp_tests/test_ecapa_speaker_embedding.py`

**Interfaces:**
- Produces: `prepare_ecapa_batch(audios) -> PreparedSpeakerBatch` and `ECAPARuntime.embed(batch) -> np.ndarray[float32]` with unit-norm 192-vectors.

- [ ] **Step 1: Write failing tests for mono resampling, relative lengths, one encoder call, stable ordering, and unit norms**

```python
def test_embed_uses_one_batch_and_l2_normalizes(fake_encoder, mixed_audio):
    vectors = ECAPARuntime(fake_encoder).embed(prepare_ecapa_batch(mixed_audio))
    assert fake_encoder.calls == 1
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_ecapa_speaker_embedding.py -q`; expect missing runtime failure**
- [ ] **Step 3: Add SpeechBrain dependency through `pyproject.toml`, update lock with the repository's Nix-wrapped uv command, and implement duration sorting/padding/inference-mode validation**
- [ ] **Step 4: Rerun focused tests; expect PASS and exactly one fake encoder call per batch**
- [ ] **Step 5: Commit with `git commit -m "feat: add batched ECAPA speaker embeddings"`**

### Task 4: Parquet shard writer and embedding node

**Files:**
- Create: `src/runner/nodes/speaker_clustering/shards.py`
- Create: `src/runner/nodes/speaker_clustering/embed_node.py`
- Test: `tmp_tests/test_speaker_embedding_shards.py`

**Interfaces:**
- Consumes: one-segment `Audio` items.
- Produces: one `SpeakerEmbeddingShardRef` per bounded inference group; schema contains segment/audio IDs, duration, quality, true label, and fixed-size float16 embedding.

- [ ] **Step 1: Write a failing round-trip test asserting fixed-size 192-vector columns and no raw audio in the artifact**
- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_embedding_shards.py -q`; expect missing node failure**
- [ ] **Step 3: Implement `ECAPASpeakerEmbed` with `MICRO_BATCH`, duration sorting, accelerator lease, keep-loaded setup/teardown, atomic artifact write, cancellation, and item/seconds progress**

```python
class ECAPASpeakerEmbedNode(Node):
    NODE_TYPE = "ECAPASpeakerEmbed"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"shard": SpeakerEmbeddingShardRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=128, max_size=512)
```

- [ ] **Step 4: Rerun shard tests with non-finite and too-short cases; expect PASS with explicit rejected quality rows**
- [ ] **Step 5: Commit with `git commit -m "feat: write ECAPA embedding shards"`**

### Task 5: Durable collector and real graph verification

**Files:**
- Create: `src/runner/nodes/speaker_clustering/collect_node.py`
- Modify: `src/runner/nodes/speaker_clustering/__init__.py`
- Modify: `src/runner/nodes/registry.py`
- Create: `workflows/speaker_embedding_ecapa.json`
- Test: `tmp_tests/test_speaker_embedding_collection.py`

**Interfaces:**
- Consumes: `SpeakerEmbeddingShardRef` stream.
- Produces: exactly one sealed `SpeakerEmbeddingSetRef` when durable row count equals expected count.

- [ ] **Step 1: Write a failing out-of-order/duplicate shard collector test**
- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_embedding_collection.py -q`; expect missing collector failure**
- [ ] **Step 3: Implement transactional register-and-seal, registration, and example workflow**
- [ ] **Step 4: Run all temporary embedding tests and `nix develop --command python -m compileall -q src`; expect PASS, then remove temporary tests with `apply_patch`**
- [ ] **Step 5: Submit the example through `POST /graphs/runs`, inspect with the Nix-wrapped CLI, and verify multiple inputs produce batched calls and a sealed shard manifest**
- [ ] **Step 6: Commit with `git commit -m "feat: collect scalable speaker embedding sets"`**
