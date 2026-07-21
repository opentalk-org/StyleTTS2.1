# Beetle Random Validation Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select a fixed, seeded random set of full validation recordings from any configured dataset instead of requiring hand-picked audio IDs.

**Architecture:** Build stage-aware validation candidate pools while constructing the existing database index, then deterministically sample ordered audio IDs from those pools during run preparation. Keep full-WAV loading, reconstruction, loss evaluation, and artifact trimming unchanged.

**Tech Stack:** Python 3.12, Pydantic, PyTorch data structures, shared PostgreSQL CRUD facades, Nix development shell.

## Global Constraints

- Use `validation.sample_count`; do not retain an explicit-ID compatibility form.
- Sample without replacement from sorted candidates using a seed derived from `runtime.seed` and the stage.
- Keep the selected set fixed for the run and reproducible on resume.
- Stage 2/3 selection must guarantee at least two distinct voices.
- Validation must continue to reconstruct complete stored recordings, independent of `adversarial.segment_samples`.
- Temporary tests live outside the repository and are removed before completion.
- Keep every source file below 300 lines and every folder at or below 16 files.

---

### Task 1: Stage-aware validation candidates and deterministic sampling

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/index.py`
- Modify: `src/runner/nodes/training/beetle/data/validation.py`
- Modify: `src/runner/nodes/training/beetle/data/validation_types.py`
- Modify: `src/runner/nodes/training/beetle/data/__init__.py`
- Test: `/tmp/beetle_random_validation_tests.py`

**Interfaces:**
- Produces: `ValidationCandidates(stage1: tuple[UUID, ...], conditional_by_voice: dict[str, tuple[UUID, ...]])`.
- Produces: `select_validation_audio_ids(index: DatabaseSegmentIndex, stage_number: int, sample_count: int, runtime_seed: int) -> tuple[UUID, ...]`.
- Consumes: `derive_seed(runtime_seed, stage_number, "validation-recordings")`.

- [ ] **Step 1: Write failing candidate and sampling tests**

Create `/tmp/beetle_random_validation_tests.py` with fixtures containing stored and external recordings, incomplete conditional metadata, single-voice and multi-voice candidates, and stable UUID ordering. Assert:

```python
first = select_validation_audio_ids(index, 1, 3, 1234)
second = select_validation_audio_ids(index, 1, 3, 1234)
assert first == second
assert len(first) == len(set(first)) == 3
assert select_validation_audio_ids(index, 1, 3, 5678) != first

conditional = select_validation_audio_ids(index, 2, 3, 1234)
voices = {index.validation.voice_for(audio_id) for audio_id in conditional}
assert len(voices) >= 2
```

Also assert exact `ValueError` messages for an unsupported stage, insufficient candidate count, and insufficient conditional voice diversity.

- [ ] **Step 2: Run the temporary tests and verify RED**

Run:

```bash
nix develop --command python /tmp/beetle_random_validation_tests.py
```

Expected: import failure for `ValidationCandidates` or `select_validation_audio_ids`, proving the behavior is absent.

- [ ] **Step 3: Add candidate pools to the index**

In `data/validation_types.py`, group the already-loaded, selection-filtered references by audio ID before eligibility filtering. Build sorted Stage 1 candidates from complete packed, non-virtual recordings. Build conditional groups only when every segment has configured language, nonempty text and phonemes, and one consistent nonempty voice ID. In `data/index.py`, attach those candidates to `DatabaseSegmentIndex` and add their IDs and voice grouping to the index fingerprint so resume rejects changed validation populations. Keeping the model and builder in `validation_types.py` prevents `index.py` from exceeding 300 lines without adding a seventeenth file to `data/`.

Use a frozen dataclass rather than raw structured dictionaries:

```python
@dataclass(frozen=True)
class ValidationCandidates:
    stage1: tuple[UUID, ...]
    conditional_by_voice: dict[str, tuple[UUID, ...]]

    def for_stage(self, stage_number: int) -> tuple[UUID, ...]:
        if stage_number == 1:
            return self.stage1
        if stage_number not in (2, 3):
            raise ValueError("validation stage number must be 1, 2, or 3")
        return tuple(
            sorted(
                audio_id
                for audio_ids in self.conditional_by_voice.values()
                for audio_id in audio_ids
            )
        )

    def voice_for(self, audio_file_id: UUID) -> str:
        matches = tuple(
            voice_id
            for voice_id, audio_ids in self.conditional_by_voice.items()
            if audio_file_id in audio_ids
        )
        if len(matches) != 1:
            raise KeyError(f"conditional validation audio is not indexed: {audio_file_id}")
        return matches[0]
```

- [ ] **Step 4: Implement deterministic selection**

In `data/validation.py`, add `select_validation_audio_ids`. Seed `random.Random` with `derive_seed(runtime_seed, stage_number, "validation-recordings")`, shuffle a sorted candidate list, and take `sample_count` items. For Stages 2/3, replace the final selected item with the first shuffled remaining candidate having a different voice when the initial prefix lacks diversity. Fail before loading WAV bytes when counts or voice diversity are insufficient.

- [ ] **Step 5: Run the temporary tests and verify GREEN**

Run:

```bash
nix develop --command python /tmp/beetle_random_validation_tests.py
```

Expected: all deterministic, seed-sensitive, eligibility, count, and voice-diversity assertions pass.

- [ ] **Step 6: Commit candidate selection**

```bash
git add src/runner/nodes/training/beetle/data/index.py \
  src/runner/nodes/training/beetle/data/validation.py \
  src/runner/nodes/training/beetle/data/validation_types.py \
  src/runner/nodes/training/beetle/data/__init__.py
git commit -m "feat: sample Beetle validation recordings"
```

### Task 2: Configuration and runtime integration

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/validation.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`
- Modify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml` (ignored local run config)
- Modify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml` (ignored active run config)
- Test: `/tmp/beetle_random_validation_config_tests.py`

**Interfaces:**
- Consumes: `select_validation_audio_ids(index, stage_number, config.validation.sample_count, config.runtime.seed)`.
- Produces: `ValidationConfig(sample_count: int = Field(gt=0))`.

- [ ] **Step 1: Write failing configuration/runtime tests**

Create `/tmp/beetle_random_validation_config_tests.py`. Assert that `ValidationConfig(sample_count=16)` passes, zero fails, and `audio_file_ids` is rejected. Exercise a factored preparation helper with a synthetic index and a fake loader to assert that runtime passes the sampled IDs to `ValidationLoader.load_source` in deterministic order.

- [ ] **Step 2: Run the tests and verify RED**

```bash
nix develop --command python /tmp/beetle_random_validation_config_tests.py
```

Expected: `sample_count` is rejected because the current model still requires `audio_file_ids`.

- [ ] **Step 3: Replace explicit IDs and integrate selection**

Replace `ValidationConfig.audio_file_ids` and its uniqueness validator with:

```python
class ValidationConfig(StrictConfigModel):
    sample_count: int = Field(gt=0)
```

Set `validation.sample_count: 16` in the default and both local Stage 1 configs. In `prepare_run`, call `select_validation_audio_ids` after index construction and pass its result to `ValidationLoader.load_source`.

- [ ] **Step 4: Run configuration/runtime tests and parse both configs**

```bash
nix develop --command python /tmp/beetle_random_validation_config_tests.py
nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; load_config("src/runner/nodes/training/beetle/config/default.yaml"); load_config("src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml"); print("configs_ok")'
```

Expected: temporary tests pass and output contains `configs_ok`.

- [ ] **Step 5: Commit configuration and runtime integration**

The local run configurations are ignored and remain uncommitted.

```bash
git add src/runner/nodes/training/beetle/config/validation.py \
  src/runner/nodes/training/beetle/config/default.yaml \
  src/runner/nodes/training/beetle/training/runtime.py
git commit -m "feat: configure random Beetle validation"
```

### Task 3: Documentation and live full-recording verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/README.md`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Verify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-harmonic-source-fix/`

**Interfaces:**
- Documents: positive sample count, fixed seeded selection, stage eligibility, strict failures, and full-recording artifacts.

- [ ] **Step 1: Update user-facing contracts**

Replace explicit ordered-ID documentation with seeded random sampling from the configured dataset. State that selection is fixed within a run, Stage 2/3 require voice diversity, and the 9,600-sample adversarial window never limits validation reconstruction.

- [ ] **Step 2: Run static verification**

```bash
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
! rg -n 'validation\.audio_file_ids|nil UUID' \
  src/runner/nodes/training/beetle/config \
  src/runner/nodes/training/beetle/data \
  src/runner/nodes/training/beetle/training \
  src/runner/nodes/training/beetle/README.md \
  src/runner/nodes/training/beetle/main.md
git diff --check
```

Expected: compileall and whitespace checks exit zero and the stale-contract search returns no matches.

- [ ] **Step 3: Verify file and folder constraints**

```bash
test "$(find src/runner/nodes/training/beetle/data -maxdepth 1 -type f | wc -l)" -le 16
find src/runner/nodes/training/beetle \
  \( -path '*/external' -o -path '*/papers' -o -path '*/runs' \) -prune -o \
  -type f \( -name '*.py' -o -name '*.md' \) -print0 | xargs -0 wc -l | \
  awk '$1 > 300 && $2 != "total" {failed=1; print} END {exit failed}'
```

Expected: both commands exit zero with no oversized owned files.

- [ ] **Step 4: Commit documentation**

```bash
git add src/runner/nodes/training/beetle/README.md src/runner/nodes/training/beetle/main.md
git commit -m "docs: describe random Beetle validation"
```

- [ ] **Step 5: Apply the configuration to training safely**

Inspect the active optimizer step and latest checkpoint. Because changing
`ValidationConfig` changes the config fingerprint, do not claim that the old
checkpoint can resume until a real resume attempt verifies it. Preserve every
existing checkpoint and output. If strict resume rejects the changed fingerprint,
start a separate fresh output directory rather than modifying checkpoint state.

- [ ] **Step 6: Verify a real validation event**

Run Stage 1 through its public standalone entrypoint as `user`, wait for a
validation event, and compare every emitted `gt.wav` frame count with the
selected database record duration. Confirm at least one artifact exceeds 9,600
samples and that the 16 selected IDs are reproduced by a second preparation with
the same seed.

- [ ] **Step 7: Remove temporary tests and run final verification**

```bash
rm /tmp/beetle_random_validation_tests.py /tmp/beetle_random_validation_config_tests.py
git status --short
git log -5 --oneline
```

Expected: no temporary tests remain, only intended ignored runtime files differ,
and the feature commits are present.
