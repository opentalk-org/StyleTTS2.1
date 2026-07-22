# Beetle Sequence Flow Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-token flow time/step sampling with per-sequence sampling and start a fresh run using only the step-8000 acoustic reconstruction weights.

**Architecture:** Keep independent `[B,C,T]` Gaussian noise, but sample flow case, step index, and time as `[B,1,1]` and broadcast them through the temporal mask. Use a temporary launch-only initialization path to copy four named checkpoint states into freshly constructed models without restoring any training state.

**Tech Stack:** Python, PyTorch, Pydantic, Nix development shell

## Global Constraints

- Run Python and tests through `nix develop --command`.
- Preserve the user's current edits in `training/loop.py` and `training/loop_events.py`.
- Do not retain a committed test or the temporary selective-initialization launcher.
- Start the replacement training process as the non-root `user` account.

---

### Task 1: Sequence-level temporal flow sampling

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/latent_flow/model.py`
- Temporary test: `/tmp/test_beetle_sequence_flow_sampling.py`

**Interfaces:**
- Consumes: `sample_flow_training_case(latent, mask, minimum_steps, base_case_probability, generator)`
- Produces: the same `FlowTrainingSample` API with temporally constant `time`, `step`, and `step_index` per batch item

- [ ] **Step 1: Write the failing temporary test**

Use this direct behavior test:

```python
import torch

from runner.nodes.training.beetle.models.modules.latent_flow.model import (
    sample_flow_training_case,
)

latent = torch.randn(4, 3, 16)
mask = torch.ones(4, 1, 16, dtype=torch.bool)
sample = sample_flow_training_case(
    latent, mask, 128, 0.5, torch.Generator().manual_seed(41)
)
for values in (sample.time, sample.step, sample.step_index):
    assert torch.all(values == values[:, :, :1])
assert torch.any(sample.noise[:, :, 1:] != sample.noise[:, :, :-1])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nix develop --command python /tmp/test_beetle_sequence_flow_sampling.py`

Expected: assertion failure because current sampling varies over time.

- [ ] **Step 3: Implement the minimal temporal sampling change**

Use `sample_shape = (latent.shape[0], 1, 1)` for scalar randomness. Enforce
mixed cases by changing the first or last valid batch item when the batch has
at least two valid items. Expand `time`, `step`, and `step_index` with
`expand_as(mask)` and apply the numeric/boolean mask before constructing the
state.

- [ ] **Step 4: Run the temporary test to verify it passes**

Run: `nix develop --command python /tmp/test_beetle_sequence_flow_sampling.py`

Expected: exit code 0.

### Task 2: Selective step-8000 initialization and restart

**Files:**
- Temporary launcher: `/tmp/start_beetle_sequence_training.py`
- Read: `src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/checkpoints/checkpoint_fde27fde4cf342589d10ea7510f3110d/payload.pt`
- Create runtime output: `src/runner/nodes/training/beetle/runs/libritts-sequence-flow/output/`

**Interfaces:**
- Consumes: checkpoint named states `audio_encoder`, `feature_linear`, `decoder`, and `generator`
- Produces: a fresh training process whose other modules and all training state start from initialization

- [ ] **Step 1: Implement a temporary launcher around the normal training construction**

Monkeypatch only the model-construction symbol used by the normal training
entry point, then call the normal CLI:

```python
import logging
from pathlib import Path

import torch

from runner.nodes.training.beetle.scripts.common import run_cli
from runner.nodes.training.beetle.training.execution import training

CHECKPOINT = Path(
    "/workspace/styletts_studio_v2/src/runner/nodes/training/beetle/runs/"
    "libritts-aligned-window-8s/output/checkpoints/"
    "checkpoint_fde27fde4cf342589d10ea7510f3110d/payload.pt"
)
NAMES = ("audio_encoder", "feature_linear", "decoder", "generator")
original_builder = training.build_acoustic_models
payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
saved = {state.name: state.value for state in payload.states if state.name in NAMES}

def initialized_builder(config, f0_extractor):
    models = original_builder(config, f0_extractor)
    for name in NAMES:
        getattr(models, name).load_state_dict(saved[name])
    logging.getLogger(__name__).info("initialized acoustic modules: %s", ", ".join(NAMES))
    return models

training.build_acoustic_models = initialized_builder
run_cli()
```

The concrete checkpoint path in the temporary script must be absolute. Because
the normal CLI receives no `--resume`, optimizer, scheduler, sampler, loop, and
reporting state remain fresh.

- [ ] **Step 2: Stop the current process cleanly**

Send `SIGINT` to the current Python training PID and wait for its exact-boundary
checkpoint/exit behavior from the user's existing loop changes.

- [ ] **Step 3: Start the replacement process as `user`**

Launch through `nix develop --command` with the same service environment and a
new output directory, retaining the process log in that directory.

- [ ] **Step 4: Verify initialization and advancement**

Inspect the log for an explicit four-module initialization report, fresh step
numbering, and at least one completed optimizer step. Confirm the process is
alive and no checkpoint compatibility fallback occurred.

### Task 3: Final verification and cleanup

**Files:**
- Remove: `/tmp/test_beetle_sequence_flow_sampling.py`
- Remove after launch: `/tmp/start_beetle_sequence_training.py`

**Interfaces:**
- Consumes: running process and repository diff
- Produces: only the requested temporal source-code change plus documentation commits

- [ ] **Step 1: Run source validation**

Run compile/import validation for the changed module through Nix and rerun the
temporary behavioral test immediately before deleting it.

- [ ] **Step 2: Inspect repository changes**

Confirm the flow sampler is the only production-code modification made here and
the user's pre-existing loop edits remain untouched.

- [ ] **Step 3: Remove temporary scripts**

Delete both `/tmp` scripts after the new process has loaded them and verified
training advancement.
