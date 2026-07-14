# Speaker ANN Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap unknown speakers from sealed ECAPA shards and produce conservative, auditable assignments at 5–10 million segment scale.

**Architecture:** Build a deterministic FAISS IVF candidate index, query in bounded blocks, exact-rerank candidates from canonical vectors, then form reciprocal high-confidence microclusters and consolidate prototypes only with multiple supporting pairs. Every segment receives accepted, provisional, ambiguous, or rejected status with scores and reasons.

**Tech Stack:** Python 3.12, FAISS, NumPy memmaps, PyArrow/Parquet, SQLAlchemy/PostgreSQL, runflow nodes, existing artifact CRUD

## Global Constraints

- ANN proposes candidates only; exact float32 cosine decides edges and assignments.
- No dense all-pairs matrix and no Python list containing the full corpus.
- Thresholds are explicit/calibrated and every decision records best score, second score, margin, candidates, version, and reason.
- One similar pair cannot merge two established clusters; require reciprocal and multi-pair support.
- Commands run through Nix; temporary tests/fixtures are removed; files stay below 300 lines.

---

### Task 1: Clustering run persistence and output contracts

**Files:**
- Modify: `src/shared/db/speakers/models.py`
- Modify: `src/shared/db/speakers/schemas.py`
- Modify: `src/shared/db/speakers/crud.py`
- Create: `migrations/versions/20260714_02_add_speaker_clustering_runs.py`
- Modify: `src/runner/nodes/models.py`
- Modify: `src/runner/nodes/datatypes.py`
- Test: `tmp_tests/test_speaker_cluster_storage.py`

**Interfaces:**
- Consumes: `SpeakerEmbeddingSetRef(run_id, artifact_ids, dimension, item_count, model_revision, preprocessing_version)`.
- Produces: `SpeakerClusterRunRef(run_id, embedding_run_id, assignment_artifact_ids, prototype_artifact_id, index_artifact_id)` and port.
- Produces CRUD for clustering runs, ordered artifact registration, cluster summaries, and atomic completion.

- [ ] **Step 1: Write a failing CRUD test proving duplicate artifact registration is idempotent and completion requires assignment count equality**
- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_cluster_storage.py -q`; expect missing models/functions**
- [ ] **Step 3: Add revision `20260714_02` with down revision `20260714_01`, typed status/outcome enums, constraints, CRUD, dataclass, and port**

```python
def complete_clustering_run(session: Session, run_id: UUID,
    assignment_count: int, prototype_artifact_id: UUID, index_artifact_id: UUID) -> SpeakerClusteringRun: ...
```

- [ ] **Step 4: Upgrade and rerun test; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: add speaker clustering run storage"`**

### Task 2: Canonical shard reader and FAISS candidate index

**Files:**
- Create: `src/runner/nodes/speaker_clustering/shard_reader.py`
- Create: `src/runner/nodes/speaker_clustering/faiss_index.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tmp_tests/test_speaker_faiss_index.py`

**Interfaces:**
- Produces: `iter_embedding_blocks(set_ref, block_rows) -> Iterator[EmbeddingBlock]` and `SpeakerCandidateIndex.train/add/search/save`.

- [ ] **Step 1: Write failing tests comparing ANN results to exact `IndexFlatIP`, deterministic sampling, and bounded block sizes**

```python
def test_candidate_index_recalls_exact_neighbors(embedding_set):
    index = build_candidate_index(embedding_set, FaissIndexSettings.for_test())
    assert recall_at_k(index.search(QUERIES), exact_search(QUERIES), k=8) >= 0.95
```

- [ ] **Step 2: Run the focused test; expect missing module/dependency failure**
- [ ] **Step 3: Add FAISS dependency with Nix-wrapped uv and implement normalized inner-product IVF training/add/search in blocks, stable int64 row IDs, serialization, cancellation callbacks, and CPU/GPU ownership rules**
- [ ] **Step 4: Rerun tests for Flat test profile and IVF production profile; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: build sharded speaker candidate index"`**

### Task 3: Exact reranking and reciprocal sparse edges

**Files:**
- Create: `src/runner/nodes/speaker_clustering/candidates.py`
- Create: `src/runner/nodes/speaker_clustering/edge_shards.py`
- Test: `tmp_tests/test_speaker_candidate_edges.py`

**Interfaces:**
- Consumes: ANN neighbor blocks plus canonical float16 embeddings.
- Produces: sorted Parquet edge shards `(left_id, right_id, exact_score, reciprocal_rank)` containing only reciprocal candidates above explicit threshold.

- [ ] **Step 1: Write a failing test where an ANN false positive is removed by exact cosine and a one-way neighbor is rejected**
- [ ] **Step 2: Run `nix develop --command pytest tmp_tests/test_speaker_candidate_edges.py -q`; expect missing functions**
- [ ] **Step 3: Implement batched float32 reranking, self-edge removal, reciprocal lookup through disk-backed sorted blocks, deterministic ordering, and bounded edge shard writes**

```python
def accepted_edge(left: RankedCandidate, reverse: RankedCandidate,
                  exact_score: float, threshold: float) -> bool: ...
```

- [ ] **Step 4: Rerun with shuffled queries and assert byte-stable edge ordering; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: exact-rerank reciprocal speaker edges"`**

### Task 4: Conservative microclusters and prototype consolidation

**Files:**
- Create: `src/runner/nodes/speaker_clustering/microclusters.py`
- Create: `src/runner/nodes/speaker_clustering/prototypes.py`
- Test: `tmp_tests/test_speaker_microclusters.py`

**Interfaces:**
- Produces disk-backed cluster labels, FP32 normalized prototype sums/counts, exemplars, dispersion, and suspicious-cluster flags.

- [ ] **Step 1: Write failing tests for chain prevention, two supporting-pair merge, one-pair merge rejection, and oversized/high-dispersion rejection**

```python
def test_single_bridge_cannot_merge_established_clusters():
    result = consolidate(CLUSTERS, support_edges=[edge("a1", "b1", 0.99)], min_support_pairs=3)
    assert result.cluster_of("a1") != result.cluster_of("b1")
```

- [ ] **Step 2: Run the focused test; expect missing clustering module**
- [ ] **Step 3: Implement union-find memmaps for seed components, streamed FP32 prototype aggregation, reciprocal prototype candidates, distinct-member support counting, exemplar retention, and explicit suspicious outcomes**
- [ ] **Step 4: Rerun tests including fixed-seed/shuffled-order reproducibility; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: build conservative speaker microclusters"`**

### Task 5: Three-way assignment and artifact output

**Files:**
- Create: `src/runner/nodes/speaker_clustering/assignment.py`
- Create: `src/runner/nodes/speaker_clustering/cluster_artifacts.py`
- Test: `tmp_tests/test_speaker_assignment_policy.py`

**Interfaces:**
- Produces assignment Parquet rows with outcome, cluster ID, best/second scores, margin, candidate IDs, threshold version, quality flags, and reason; prototype and cluster-summary artifacts.

- [ ] **Step 1: Write failing boundary tests for accepted, provisional-new, ambiguous, and quality-rejected outcomes**

```python
assert decide(scores(0.82, 0.61), policy(accept=0.80, margin=0.10, new=0.55)).outcome is ACCEPTED
assert decide(scores(0.70, 0.66), policy(accept=0.80, margin=0.10, new=0.55)).outcome is AMBIGUOUS
```

- [ ] **Step 2: Run test; expect missing policy**
- [ ] **Step 3: Implement typed policy with inclusive boundary semantics, cluster-dispersion adjustment, high-confidence-only prototype updates, bounded artifact writers, and counts by outcome**
- [ ] **Step 4: Rerun tests and verify ambiguous rows never update prototypes; expect PASS**
- [ ] **Step 5: Commit with `git commit -m "feat: assign speakers with explicit rejection"`**

### Task 6: Clustering node and scale verification

**Files:**
- Create: `src/runner/nodes/speaker_clustering/cluster_node.py`
- Modify: `src/runner/nodes/speaker_clustering/__init__.py`
- Modify: `src/runner/nodes/registry.py`
- Create: `workflows/speaker_ann_cluster.json`
- Test: `tmp_tests/test_speaker_cluster_node.py`

**Interfaces:**
- Consumes: one sealed `SpeakerEmbeddingSetRef`.
- Produces: one completed `SpeakerClusterRunRef`.

- [ ] **Step 1: Write a failing end-to-end node test over synthetic separable embeddings plus deliberately close impostors**
- [ ] **Step 2: Run focused test; expect missing node/registration**
- [ ] **Step 3: Implement `ClusterSpeakerEmbeddings` stage orchestration, explicit settings, resource policy, artifact/CRUD transitions, cancellation cleanup, progress, registration, and example graph**

```python
class ClusterSpeakerEmbeddingsNode(Node):
    NODE_TYPE = "ClusterSpeakerEmbeddings"
    INPUTS = {"embeddings": SpeakerEmbeddingSetRefPort()}
    OUTPUTS = {"cluster_run": SpeakerClusterRunRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
```

- [ ] **Step 4: Run all temporary clustering tests and compileall; expect PASS, then remove temporary tests with `apply_patch`**
- [ ] **Step 5: Generate 1,000,000 synthetic vectors in bounded Parquet shards, run the real graph through `POST /graphs/runs`, inspect CLI logs/RSS/cancellation, and remove the fixture**
- [ ] **Step 6: Commit with `git commit -m "feat: cluster speaker embeddings at scale"`**
