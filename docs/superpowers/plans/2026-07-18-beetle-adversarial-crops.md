# Beetle Adversarial Crops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Beetle's decoder, generator, reconstruction loss, and StyleTTS discriminators on aligned 9,600-sample crops while all upstream objectives retain full utterances.

**Architecture:** A model-owned `AlignedSegments` value selects matching full-rate frames, half-rate latents, and waveform samples. Stage 1 and Stage 3 build deterministic plans from checkpointed loop coordinates, run full encoders and acoustic prediction, and apply the plan only before decoder synthesis and waveform losses. Stage 2 and full-utterance validation do not use segments.

**Tech Stack:** Python 3.12, PyTorch, Pydantic, YAML, Nix.

## Global Constraints

- Do not use subagents.
- Run Python only through `nix develop --command python ...`.
- Keep every file below 300 lines and every folder at no more than 16 files.
- Keep temporary tests outside the repository and remove them before completion.
- Stage 1 and Stage 3 train the existing StyleTTS discriminators; Stage 2 does not.
- The crop is exactly 9,600 samples, 32 hop-300 frames, and 16 half-rate latent frames.
- Validation remains full-utterance.

---

### Task 1: Strict segment geometry

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/__init__.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml`
- Test: `/tmp/beetle_adversarial_crop_tests/test_config.py`

**Interfaces:**
- Produces: `AdversarialConfig(segment_samples: int)` and `BeetleConfig.adversarial`.
- Enforces: samples divide evenly by `audio.hop_length`; resulting frames divide evenly by `architecture.posterior.downsample_rate`.

- [ ] **Step 1: Write the failing configuration check**

```python
from copy import deepcopy
import yaml

from runner.nodes.training.beetle.config import BeetleConfig

payload = yaml.safe_load(open("src/runner/nodes/training/beetle/config/default.yaml"))
assert payload["adversarial"]["segment_samples"] == 9600
config = BeetleConfig.model_validate(payload)
assert config.adversarial.segment_samples == 9600
invalid = deepcopy(payload)
invalid["adversarial"]["segment_samples"] = 9500
try:
    BeetleConfig.model_validate(invalid)
except ValueError as error:
    assert "segment_samples" in str(error)
else:
    raise AssertionError("misaligned segment_samples was accepted")
```

- [ ] **Step 2: Run RED**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_config.py`

Expected: failure because `adversarial` is absent.

- [ ] **Step 3: Add the strict config model and explicit YAML field**

```python
class AdversarialConfig(StrictConfigModel):
    segment_samples: int = Field(gt=0)
```

Add `adversarial: {segment_samples: 9600}` in the same explicit top-level order in both YAML files and validate its hop/downsample geometry in `BeetleConfig.validate_composition()`.

- [ ] **Step 4: Run GREEN**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_config.py`

Expected: exit 0.

### Task 2: Aligned tensor segments

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/segments.py`
- Test: `/tmp/beetle_adversarial_crop_tests/test_segments.py`

**Interfaces:**
- Produces: `AlignedSegments.random(frame_mask, frame_count, frame_alignment, sample_hop, generator)`.
- Produces: `frames(values)`, `latents(values)`, and `samples(values)` preserving the batch and channel axes.

- [ ] **Step 1: Write failing geometry and determinism checks**

```python
import torch

from runner.nodes.training.beetle.models.modules.segments import AlignedSegments

mask = torch.arange(40).view(1, 1, 40) < torch.tensor([[[40]], [[36]]])
first = AlignedSegments.random(mask, 32, 2, 300, torch.Generator().manual_seed(7))
second = AlignedSegments.random(mask, 32, 2, 300, torch.Generator().manual_seed(7))
assert torch.equal(first.frame_starts, second.frame_starts)
assert torch.all(first.frame_starts % 2 == 0)
assert first.frames(torch.arange(80).reshape(2, 40)).shape == (2, 32)
assert first.latents(torch.zeros(2, 192, 20)).shape == (2, 192, 16)
assert first.samples(torch.zeros(2, 1, 12000)).shape == (2, 1, 9600)
```

- [ ] **Step 2: Run RED**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_segments.py`

Expected: import failure because `segments.py` is absent.

- [ ] **Step 3: Implement aligned selection**

Use vectorized `torch.gather` on the final dimension. Derive each example's valid aligned start count from `frame_mask.sum(-1)`, reject any item shorter than `frame_count`, and select starts with the supplied generator. Never select padded frames.

- [ ] **Step 4: Run GREEN**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_segments.py`

Expected: exit 0.

### Task 3: Cropped Stage 1 and Stage 3 synthesis

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/stages.py`
- Test: `/tmp/beetle_adversarial_crop_tests/test_training_contract.py`

**Interfaces:**
- Produces: `Stage1Models.reconstruct_segment(..., segment: AlignedSegments, ...)`.
- Stage 1 synthesis retains full `AudioPosterior` and `AcousticFeatures`, but returns 32-frame decoder output and 9,600 generated samples.
- Stage 3 shares one segment between posterior, conditional, and real waveforms per pass.

- [ ] **Step 1: Write the failing training contract check**

Construct reduced real Stage 1 modules, wrap the audio encoder and FeatureLinear to record input/output lengths, call segmented reconstruction on a 40-frame batch, and assert full lengths `40/20/40` before decoder lengths `16/32` and waveform length `9600`. Add source inspection assertions that Stage 3 passes the same `AlignedSegments` instance to posterior and conditional paths and crops real audio before both GAN losses.

- [ ] **Step 2: Run RED**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_training_contract.py`

Expected: failure because segmented reconstruction is absent.

- [ ] **Step 3: Implement the crop boundary**

Refactor `Stage1Models` so full reconstruction and segmented reconstruction share full encoder/FeatureLinear work, then differ only at decoder input selection. In trainers, derive segment generators with `derive_seed(runtime_seed, stage, cycle, batch_index, view, "segment")`. Use complete masks for KL/F0/noise and cropped real/generated tensors for reconstruction/GAN objectives.

- [ ] **Step 4: Run GREEN and line limits**

Run: `nix develop --command python /tmp/beetle_adversarial_crop_tests/test_training_contract.py`

Run: `wc -l src/runner/nodes/training/beetle/models/model.py src/runner/nodes/training/beetle/training/stage1.py src/runner/nodes/training/beetle/training/stage3.py`

Expected: checks pass and every file is at most 300 lines.

### Task 4: Static compilation and documentation

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/compilation.py`
- Modify: `src/runner/nodes/training/beetle/README.md`
- Modify: `src/runner/nodes/training/beetle/main.md`

**Interfaces:**
- `compile_stage1()` compiles fixed-shape AudioEncoder, FeatureLinear, and Decoder modules without dynamic shape lowering. Generator remains eager because its complex harmonic-phase cumulative path fails TorchInductor lowering.

- [ ] **Step 1: Replace dynamic module compilation**

Change supported modules from dynamic to static compilation and exclude Generator after reproducing its static and dynamic TorchInductor lowering failure.

- [ ] **Step 2: Document the approved crop contract**

Document full-utterance upstream objectives, the 9,600-sample adversarial segment, Stage 1/3 ownership, deterministic resume, and full-utterance validation in both model documents.

- [ ] **Step 3: Verify syntax and stale statements**

Run: `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`

Run: `rg -n "9600|9,600|full.utterance|segment_samples" src/runner/nodes/training/beetle/README.md src/runner/nodes/training/beetle/main.md src/runner/nodes/training/beetle/config/default.yaml`

Expected: compilation succeeds and all three files state the same geometry.

### Task 5: Fresh batch-64 training verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml`
- Remove: `/tmp/beetle_adversarial_crop_tests/`

**Interfaces:**
- Launches Stage 1 at `batch_size: 64`, `accumulation_steps: 1`, without `--resume`.

- [ ] **Step 1: Restore batch 64 and clean generated runs**

Set the local Stage 1 configuration to `64 × 1`. Stop only the Beetle training process, then remove every generated child of `runs/ljspeech-stage1/` except `config.yaml`.

- [ ] **Step 2: Launch through Nix with the canonical local service environment**

Run in `beetle-stage1:0.0`:

```bash
RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow \
MLFLOW_TRACKING_URI=http://127.0.0.1:7860 \
nix develop --command python -m runner.nodes.training.beetle.scripts.train_stage1 \
  --config src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml \
  --output src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-compiled
```

- [ ] **Step 3: Verify the real run**

Require logs for `step=0` discriminator completion, generator completion with all seven Stage 1 losses, optimizer completion with both learning rates and discriminator gradient norm, and continued progress beyond step 1. Query MLflow to confirm the adversarial metrics are stored.

- [ ] **Step 4: Remove temporary checks and commit**

Delete `/tmp/beetle_adversarial_crop_tests/`, run `git diff --check`, confirm `git status --short` contains only intended source/docs changes, and commit the implementation.
