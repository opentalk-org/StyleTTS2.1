# Beetle MLflow Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Do not use
> subagents for this repository task.

**Goal:** Replace over-nested Beetle MLflow metric names with shallow groups,
remove per-sample validation time series, and use one-based validation artifact
folders without UUID suffixes.

**Architecture:** Metric producers emit names that already match their owning
group, while the reporter adds only the `train/` prefix to raw loss names.
Validation retains per-sample data solely in its manifest and publishes only
aggregate metrics. Artifact ordering remains defined by the explicit validation
configuration and is represented by one-based sample positions.

**Tech Stack:** Python 3.12, PyTorch, MLflow, pytest, Nix development shell.

## Global Constraints

- Loss metrics use only `train/<name>` and aggregate `validation/<name>`.
- There are no epoch metrics and no per-sample MLflow metrics.
- Metric names have at most one slash.
- Artifact directories are `sample_1`, `sample_2`, and so on.
- `metrics.json` retains one-based position, UUID, seed, losses, and artifacts.
- Temporary tests are removed before completion.
- All Python commands run through `nix develop --command`.
- Do not restart the active Stage 1 training process while changing code.

---

### Task 1: Shallow MLflow metric names

**Files:**

- Create temporarily: `src/runner/nodes/training/beetle/_metric_names_test.py`
- Modify: `src/runner/nodes/training/beetle/training/reporting/reporter.py`
- Modify: `src/runner/nodes/training/beetle/training/optimizer.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/runtime.py`

**Interfaces:**

- Consumes: raw `TrainingMetric(name, value)` values from stage trainers.
- Produces: shallow MLflow names and aggregate-only
  `validation_metrics(result) -> tuple[TrainingMetric, ...]`.

- [ ] **Step 1: Write the failing temporary tests**

Create tests that exercise real reporter, optimizer, and validation metric
construction. The essential assertions are:

```python
def test_reporter_uses_shallow_train_names():
    reporter.publish(observation, ReportingState.initial("run"), ())
    names = tuple(metric.name for metric in session.submitted)
    assert "train/reconstruction" in names
    assert all(name.count("/") <= 1 for name in names)


def test_optimizer_uses_shallow_names():
    metrics = optimizer_set.step(1, (NamedGradientGroup("encoder", (module,)),))
    assert tuple(metric.name for metric in metrics) == (
        "optimizer/generator_learning_rate",
        "optimizer/generator_gradient_norm",
        "optimizer/generator_amp_scale",
        "gradient/encoder",
    )


def test_validation_publishes_only_aggregate_metrics():
    metrics = validation_metrics(validation_result())
    assert tuple(metric.name for metric in metrics) == (
        "validation/reconstruction",
    )
```

- [ ] **Step 2: Run the temporary tests and verify RED**

Run:

```bash
nix develop --command pytest \
  src/runner/nodes/training/beetle/_metric_names_test.py -q
```

Expected: failures show `train/loss/reconstruction`,
`optimizer/learning_rate/generator`, `gradient_norm/optimizer/generator`, and
`validation/loss/reconstruction` plus per-sample validation names.

- [ ] **Step 3: Implement the minimal naming changes**

In `TrainingReporter.publish`, change the loss wrapper to:

```python
TrainingMetric(f"train/{metric.name}", metric.value)
```

In `ScheduledOptimizer.finish`, emit:

```python
TrainingMetric(f"optimizer/{self.name}_learning_rate", learning_rate)
TrainingMetric(f"optimizer/{self.name}_gradient_norm", float(gradient_norm))
TrainingMetric(f"optimizer/{self.name}_amp_scale", self.scaler.get_scale())
```

In `OptimizerSet.step`, emit:

```python
TrainingMetric(
    f"gradient/{group.name}",
    _gradient_norm(_unique_parameters(group.modules)),
)
```

In `validation_metrics`, return only:

```python
return tuple(
    TrainingMetric(f"validation/{metric.name}", metric.value)
    for metric in result.aggregates
)
```

- [ ] **Step 4: Run the temporary tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

---

### Task 2: One-based validation artifact folders and documentation

**Files:**

- Extend temporarily: `src/runner/nodes/training/beetle/_metric_names_test.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/artifacts.py`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `src/runner/nodes/training/beetle/README.md`

**Interfaces:**

- Consumes: ordered `ValidationResult.samples`.
- Produces: `sample_<one-based-position>` directories and a manifest retaining
  source UUID lineage.

- [ ] **Step 1: Add the failing artifact contract test**

Publish two real CPU validation samples through `ValidationArtifacts` with a
recording uploader. Assert:

```python
artifact_paths = tuple(uploader.artifact_paths)
assert any("/sample_1" in path for path in artifact_paths)
assert any("/sample_2" in path for path in artifact_paths)
assert all(str(first_uuid) not in path for path in artifact_paths)

manifest = json.loads(
    (tmp_path / "validation/stage1/step_4/metrics.json").read_text()
)
assert manifest["samples"][0]["position"] == 1
assert manifest["samples"][0]["audio_file_id"] == str(first_uuid)
assert manifest["samples"][1]["position"] == 2
```

- [ ] **Step 2: Run the artifact test and verify RED**

Run the Task 1 test command. Expected: paths contain zero-based positions and
UUID suffixes, and manifest positions begin at zero.

- [ ] **Step 3: Implement one-based short sample names**

Use one-based enumeration for both artifact jobs and manifest entries:

```python
for position, sample in enumerate(result.samples, start=1):
    sample_name = f"sample_{position}"
```

and:

```python
for position, sample in enumerate(result.samples, start=1)
```

Update `main.md` and `README.md` to list the shallow metric contract,
aggregate-only validation time series, UUID lineage in `metrics.json`, and
`sample_<one-based-position>` artifact directories.

- [ ] **Step 4: Run focused tests and repository checks**

Run:

```bash
nix develop --command pytest \
  src/runner/nodes/training/beetle/_metric_names_test.py -q
nix develop --command python -m compileall -q \
  src/runner/nodes/training/beetle
rg -n 'train/loss|validation/loss|validation/sample|gradient_norm/(optimizer|module)|sample_<position>_<audio_id>' \
  src/runner/nodes/training/beetle
```

Expected: tests and compilation pass; the search finds no stale contract in
production code or documentation.

- [ ] **Step 5: Remove temporary tests and commit**

Delete only the temporary `_metric_names_test.py`, run `git diff --check`, and
commit production code plus documentation as:

```bash
git commit -m "fix: simplify beetle training reports"
```

---

### Task 3: Live verification without restarting training

**Files:** None.

**Interfaces:**

- Consumes: the current tmux run and a short separate smoke run.
- Produces: evidence that the current process remains alive and newly started
  processes use the approved MLflow/artifact contract.

- [ ] **Step 1: Confirm the active training process still advances**

Run `tmux capture-pane -pt beetle-stage1:training -S -12`. Expected: optimizer
steps continue without interruption.

- [ ] **Step 2: Run a short isolated validation smoke**

Use the existing smoke configuration with a fresh output directory and MLflow
run. Do not restart or replace `beetle-stage1:training`.

- [ ] **Step 3: Inspect MLflow and artifacts**

Assert the fresh run contains no metric with more than one slash, no
`validation/sample` prefix, aggregate `validation/*` metrics, and artifact
directories named `sample_1` through the configured sample count.

- [ ] **Step 4: Final verification**

Run `git status --short`, `git diff --check`, and capture the active tmux pane.
Expected: a clean worktree and advancing training process.
