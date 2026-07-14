# Speaker Cluster Audit and Apply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit clustered speaker assignments for false merges, create deterministic listening evidence, and safely apply accepted assignments to stored segments.

**Architecture:** Consume a sealed `SpeakerClusterRunRef` produced by the ANN clustering plan. Stream assignment and embedding shards for sampled quantitative checks, persist compact audit artifacts, then bulk-create voices and rewrite each audio record's segment JSON once while leaving ambiguous segments unassigned.

**Tech Stack:** Python 3.12, Pydantic, SQLAlchemy/PostgreSQL, NumPy, scikit-learn metrics, PyArrow/Parquet, existing S3 asset CRUD, runflow nodes

## Global Constraints

- All Python and graph commands run through `nix develop --command ...`.
- Database access uses `src/shared/db/speakers/crud.py`, audio CRUD, voice CRUD, and asset CRUD only.
- Bulk payloads stay in artifacts; PostgreSQL stores compact audit state and artifact references.
- Temporary tests and generated corpora are removed before completion.
- Files stay below 300 lines and folders below 16 files.
- Ambiguous or rejected segments retain `voice_id=None`.

---

### Task 1: Typed audit result and persisted audit state

**Files:**
- Modify: `src/runner/nodes/models.py`
- Modify: `src/runner/nodes/datatypes.py`
- Modify: `src/shared/db/speakers/models.py`
- Modify: `src/shared/db/speakers/schemas.py`
- Modify: `src/shared/db/speakers/crud.py`
- Create: `migrations/versions/20260714_03_add_speaker_cluster_audits.py`
- Test: `tmp_tests/test_speaker_audit_storage.py`

**Interfaces:**
- Consumes: `SpeakerClusterRunRef(run_id: UUID, embedding_run_id: UUID, assignment_artifact_ids: tuple[UUID, ...])`.
- Produces: `SpeakerAuditRef(audit_id: UUID, cluster_run_id: UUID, report_artifact_id: UUID, listening_artifact_id: UUID)` and `SpeakerAuditRefPort`.

- [ ] **Step 1: Write a failing temporary CRUD round-trip test**

```python
def test_complete_audit_records_artifacts(session, cluster_run):
    audit = speaker_crud.create_audit(session, SpeakerAuditCreate(cluster_run_id=cluster_run.id, seed=7))
    completed = speaker_crud.complete_audit(session, audit.id, report_artifact_id=REPORT_ID, listening_artifact_id=LISTEN_ID, metrics={"pair_precision": 1.0})
    assert completed.metrics["pair_precision"] == 1.0
```

- [ ] **Step 2: Run the test through Nix and confirm the missing schema/CRUD failure**

Run: `nix develop --command pytest tmp_tests/test_speaker_audit_storage.py -q`
Expected: FAIL because audit models and functions do not exist.

- [ ] **Step 3: Add the migration, typed schemas, exact CRUD functions, runner dataclass, and port**

```python
@dataclass(frozen=True)
class SpeakerAuditRef:
    audit_id: UUID
    cluster_run_id: UUID
    report_artifact_id: UUID
    listening_artifact_id: UUID

def complete_audit(session: Session, audit_id: UUID, report_artifact_id: UUID,
                   listening_artifact_id: UUID, metrics: SpeakerAuditMetrics) -> SpeakerClusterAudit: ...
```

- [ ] **Step 4: Upgrade the temporary database and rerun the test**

Run: `nix develop --command alembic upgrade head && nix develop --command pytest tmp_tests/test_speaker_audit_storage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the audit persistence contract**

```bash
git add migrations/versions src/shared/db/speakers src/runner/nodes/models.py src/runner/nodes/datatypes.py
git commit -m "feat: add speaker cluster audit records"
```

### Task 2: Labeled and unlabeled audit metrics

**Files:**
- Create: `src/runner/nodes/speaker_clustering/audit_metrics.py`
- Test: `tmp_tests/test_speaker_audit_metrics.py`

**Interfaces:**
- Consumes: iterables of `AssignmentAuditRow(segment_id, cluster_id, true_label, centroid_score, second_score)` and sampled pair scores.
- Produces: `SpeakerAuditMetrics` with pair precision/recall, weighted purity, ARI, AMI, fragmentation, distinct-label maxima, score quantiles, and suspicious cluster IDs.

- [ ] **Step 1: Write failing tests with a deliberate two-speaker merge and a fragmented speaker**

```python
def test_metrics_expose_random_speaker_merge():
    rows = labeled_rows(predicted=["a", "a", "a", "b"], truth=["x", "x", "y", "y"])
    result = compute_labeled_metrics(rows)
    assert result.pair_precision < 1.0
    assert result.max_true_speakers_in_cluster == 2
    assert result.fragmented_speaker_count == 1
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `nix develop --command pytest tmp_tests/test_speaker_audit_metrics.py -q`
Expected: FAIL because `audit_metrics` does not exist.

- [ ] **Step 3: Implement bounded aggregate metrics and deterministic stratified sampling helpers**

```python
def compute_labeled_metrics(rows: Iterable[AssignmentAuditRow]) -> LabeledAuditMetrics: ...
def score_distribution(values: Iterable[float]) -> ScoreDistribution: ...
def deterministic_sample_ids(rows: Iterable[AuditSampleKey], size: int, seed: int) -> list[UUID]: ...
```

- [ ] **Step 4: Rerun metrics tests including shuffled-input determinism**

Run: `nix develop --command pytest tmp_tests/test_speaker_audit_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit metrics**

```bash
git add src/runner/nodes/speaker_clustering/audit_metrics.py
git commit -m "feat: add speaker clustering audit metrics"
```

### Task 3: Listening samples and audit node

**Files:**
- Create: `src/runner/nodes/speaker_clustering/audit_artifacts.py`
- Create: `src/runner/nodes/speaker_clustering/audit_node.py`
- Modify: `src/runner/nodes/speaker_clustering/__init__.py`
- Test: `tmp_tests/test_speaker_audit_node.py`

**Interfaces:**
- Consumes: `SpeakerClusterRunRef` and registered assignment/embedding shard paths from speaker CRUD.
- Produces: one `SpeakerAuditRef`, a JSON/HTML report, and a ZIP manifest containing worst within-cluster, closest cross-cluster, and low-margin WAV pairs.

- [ ] **Step 1: Write a failing node test with fake shard readers and audio CRUD**

```python
async def test_audit_node_exports_riskiest_pairs(node_context, cluster_ref):
    outputs = await run_node_batch(SpeakerClusterAuditNode, {"cluster_run": cluster_ref}, node_context)
    assert outputs[0]["audit"].cluster_run_id == cluster_ref.run_id
    assert stored_manifest()["groups"]["worst_within_cluster"]
```

- [ ] **Step 2: Run it and confirm registration/module failure**

Run: `nix develop --command pytest tmp_tests/test_speaker_audit_node.py -q`
Expected: FAIL because the audit node is absent.

- [ ] **Step 3: Implement streaming shard reads, exact sampled cosine checks, ANN recall comparison, artifact rendering, cancellation, and progress**

```python
class SpeakerClusterAuditNode(Node):
    NODE_TYPE = "SpeakerClusterAudit"
    INPUTS = {"cluster_run": SpeakerClusterRunRefPort()}
    OUTPUTS = {"audit": SpeakerAuditRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=False)
```

- [ ] **Step 4: Rerun node tests and assert exact-search disagreement is reported**

Run: `nix develop --command pytest tmp_tests/test_speaker_audit_node.py -q`
Expected: PASS.

- [ ] **Step 5: Commit audit node and artifacts**

```bash
git add src/runner/nodes/speaker_clustering
git commit -m "feat: audit speaker clusters and export listening samples"
```

### Task 4: Safe bulk assignment writeback

**Files:**
- Create: `src/shared/db/audio/speaker_assignment_crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Create: `src/runner/nodes/speaker_clustering/apply_node.py`
- Test: `tmp_tests/test_speaker_assignment_apply.py`

**Interfaces:**
- Consumes: audited assignment shards through `SpeakerAuditRef`.
- Produces: `SaveResult` containing accepted, ambiguous, rejected, created-voice, and updated-audio counts.

- [ ] **Step 1: Write a failing test proving one update per audio and preservation of unrelated fields**

```python
def test_bulk_apply_preserves_segment_payload(session, audio_with_two_segments):
    result = audio_crud.bulk_apply_speaker_assignments(session, assignments)
    saved = audio_crud.list_audio_segments(session, audio_with_two_segments.id)
    assert saved[0]["alignment"] == original_alignment
    assert saved[0]["voice_id"] == str(VOICE_ID)
    assert saved[1]["voice_id"] is None
    assert result.updated_audio_count == 1
```

- [ ] **Step 2: Run it and confirm the missing bulk function failure**

Run: `nix develop --command pytest tmp_tests/test_speaker_assignment_apply.py -q`
Expected: FAIL because bulk apply is absent.

- [ ] **Step 3: Implement grouped whole-record updates, bulk voice creation, and the apply node**

```python
def bulk_apply_speaker_assignments(session: Session,
    assignments: Iterable[AcceptedSpeakerAssignment]) -> SpeakerApplyCounts: ...

class ApplySpeakerClustersNode(Node):
    NODE_TYPE = "ApplySpeakerClusters"
    INPUTS = {"audit": SpeakerAuditRefPort()}
    OUTPUTS = {"save_result": SaveResultPort()}
```

- [ ] **Step 4: Rerun tests with ambiguous segments and missing segment IDs as explicit errors**

Run: `nix develop --command pytest tmp_tests/test_speaker_assignment_apply.py -q`
Expected: PASS.

- [ ] **Step 5: Commit writeback**

```bash
git add src/shared/db/audio src/runner/nodes/speaker_clustering/apply_node.py
git commit -m "feat: apply audited speaker assignments in bulk"
```

### Task 5: Registration, workflow, and end-to-end verification

**Files:**
- Modify: `src/runner/nodes/registry.py`
- Create: `workflows/speaker_cluster_ecapa.json`
- Test: `tmp_tests/test_speaker_pipeline_registration.py`

**Interfaces:**
- Consumes: all node classes and ports from the three speaker plans.
- Produces: discoverable node schemas and a runnable graph from dataset segments to applied voices.

- [ ] **Step 1: Write a failing registry/workflow contract test**

```python
def test_speaker_pipeline_is_registered():
    names = {node.NODE_TYPE for node in build_node_registry().nodes}
    assert {"SpeakerSegmentSource", "ECAPASpeakerEmbed", "CollectSpeakerEmbeddings", "ClusterSpeakerEmbeddings", "SpeakerClusterAudit", "ApplySpeakerClusters"} <= names
```

- [ ] **Step 2: Run the focused test and confirm missing registrations**

Run: `nix develop --command pytest tmp_tests/test_speaker_pipeline_registration.py -q`
Expected: FAIL until every node is registered.

- [ ] **Step 3: Register nodes and add the example graph with explicit calibration thresholds and dataset scope**

The workflow edges must follow:

```text
SpeakerSegmentSource -> ECAPASpeakerEmbed -> CollectSpeakerEmbeddings
CollectSpeakerEmbeddings -> ClusterSpeakerEmbeddings -> SpeakerClusterAudit -> ApplySpeakerClusters
```

- [ ] **Step 4: Run static checks and remove all temporary tests**

Run: `nix develop --command pytest tmp_tests/test_speaker_pipeline_registration.py -q && nix develop --command python -m compileall -q src`
Expected: PASS, then delete `tmp_tests/test_speaker_*.py` with `apply_patch`.

- [ ] **Step 5: Run a real labeled multi-speaker graph and inspect results**

Run: `nix develop --command runflow-dev-session`, submit `workflows/speaker_cluster_ecapa.json` through `POST /graphs/runs`, save the returned ID in `RUN_ID`, then run `nix develop --command python -m cli logs "$RUN_ID"` and `nix develop --command python -m cli failed "$RUN_ID"`.
Expected: the run succeeds; known same-speaker clips share accepted clusters, unrelated speakers do not, pair precision is reported, and listening artifacts exist.

- [ ] **Step 6: Run a generated large-vector fixture and verify bounded memory/cancellation**

Run: `nix develop --command python tmp_speaker_scale_fixture.py --rows 1000000 --shard-rows 50000`, submit its clustering graph, record peak RSS, cancel one run, then remove the fixture with `apply_patch`.
Expected: 20 bounded shards, no million-row Python list, clean cancellation, and no partial sealed run.

- [ ] **Step 7: Commit integration and workflow**

```bash
git add src/runner/nodes/registry.py workflows/speaker_cluster_ecapa.json
git commit -m "feat: register scalable speaker clustering workflow"
```
