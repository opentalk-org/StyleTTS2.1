# Speaker Assignment Audit Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded streaming assignment-audit report generator with deterministic listening samples and JSON/HTML artifacts.

**Architecture:** A cancellable Parquet scanner feeds the existing audit metric functions in bounded sequential passes. A separate bounded heap selector produces deterministic manifest categories, and a renderer writes typed JSON and simple HTML artifacts atomically.

**Tech Stack:** Python 3.12, PyArrow Parquet, Pydantic v2, existing `audit_metrics`, pytest.

## Global Constraints

- Change only `src/runner/nodes/speaker_clustering/audit_report/` during implementation.
- Do not edit nodes, cluster runtime, persistence, pipeline, schemas, models, databases, or temporary scale fixtures.
- Keep every file below 300 lines and the package below 16 files.
- Run every command through `nix develop --command`.
- Keep tests temporary and remove them before completion.

---

### Task 1: Typed report and bounded selection

**Files:**
- Create: `src/runner/nodes/speaker_clustering/audit_report/models.py`
- Create: `src/runner/nodes/speaker_clustering/audit_report/selection.py`
- Test: `/tmp/test_assignment_audit_report.py`

**Interfaces:**
- Consumes: assignment column values and `SpeakerAuditMetrics`.
- Produces: `ListeningEntry`, `ListeningManifest`, `AssignmentAuditDocument`, `AssignmentAuditBuildResult`, and `select_listening_manifest(rows, suspicious_cluster_ids, limit)`.

- [ ] **Step 1: Write failing tests** that construct reordered assignment dictionaries and assert each category contains at most `limit` entries in the required score order with identical serialized output.
- [ ] **Step 2: Verify RED** with `nix develop --command uv run --frozen --with pytest python -m pytest /tmp/test_assignment_audit_report.py -q`; expect import failure for `audit_report`.
- [ ] **Step 3: Implement frozen Pydantic models** with exact string IDs, duration, nullable cluster/label/score fields, tuple categories, metrics, total rows, and artifact paths.
- [ ] **Step 4: Implement deterministic heaps** where the canonical ascending ranks are `(best_score, segment_id)`, `(-second_score, segment_id)`, `(margin, segment_id)`, and `(cluster_id, true_label, segment_id)`.
- [ ] **Step 5: Verify GREEN** with the focused pytest command; expect the selection tests to pass.

### Task 2: Streaming Parquet metrics and progress

**Files:**
- Create: `src/runner/nodes/speaker_clustering/audit_report/scans.py`
- Create: `src/runner/nodes/speaker_clustering/audit_report/builder.py`
- Modify: `/tmp/test_assignment_audit_report.py`

**Interfaces:**
- Consumes: `Sequence[Path]`, positive `batch_rows`, `Callable[[], None]`, and `Callable[[int, int], None]`.
- Produces: `build_assignment_audit_report(paths, output_dir, batch_rows, category_limit, check_cancel, report_progress) -> AssignmentAuditBuildResult`.

- [ ] **Step 1: Extend failing tests** to write small assignment Parquet files, use non-UUID IDs, assert cancellation and `(processed, total)` calls per batch, and assert unevaluable metrics serialize as null.
- [ ] **Step 2: Verify RED** with the focused pytest command; expect the missing builder failure.
- [ ] **Step 3: Implement scanning** by validating required columns, summing Parquet metadata rows, calling cancellation before every yielded batch, and reporting after every processed batch.
- [ ] **Step 4: Implement bounded metric passes** using `compute_labeled_metrics` for typed assignment rows and `score_distribution` for best, second, margin, and flattened candidate score streams.
- [ ] **Step 5: Implement the manifest pass** with only required columns and the Task 1 selector, then assemble the typed document.
- [ ] **Step 6: Verify GREEN** with the focused pytest command; expect all streaming, cancellation, progress, and metric tests to pass.

### Task 3: Artifact rendering and package API

**Files:**
- Create: `src/runner/nodes/speaker_clustering/audit_report/render.py`
- Create: `src/runner/nodes/speaker_clustering/audit_report/__init__.py`
- Modify: `/tmp/test_assignment_audit_report.py`

**Interfaces:**
- Consumes: `AssignmentAuditDocument`, `ListeningManifest`, and output directory.
- Produces: atomic `audit-report.json`, `audit-report.html`, and `listening-manifest.json` plus public package exports.

- [ ] **Step 1: Extend failing tests** to assert exact filenames, valid JSON, HTML escaping, nullable metrics, and no leftover temporary files.
- [ ] **Step 2: Verify RED** with the focused pytest command; expect missing artifacts.
- [ ] **Step 3: Implement rendering** with `model_dump_json`, escaped report JSON inside minimal HTML, sibling temporary files, and `Path.replace`.
- [ ] **Step 4: Export the public builder and typed models** from the package root.
- [ ] **Step 5: Verify GREEN** with the focused pytest command; expect every test to pass.
- [ ] **Step 6: Verify constraints** using compileall, Ruff, `git diff --check`, file line counts, and a scoped diff review.
- [ ] **Step 7: Remove `/tmp/test_assignment_audit_report.py`** and commit only the new package files.
