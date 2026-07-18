# Beetle Cropped F0 Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen F0 target extractor only on the aligned audio interval supplied to the training decoder and waveform generator.

**Architecture:** `Stage1Models` exposes separate full-rate N targets and segment-scoped F0 targets. Stage 1 and Stage 3 reuse their existing `AlignedSegments` value for F0 target extraction, prediction selection, and masking; validation retains the full-utterance `acoustic_targets` path.

**Tech Stack:** Python 3.12, PyTorch, Nix, temporary executable contract checks.

## Global Constraints

- Do not use subagents.
- Run Python only through `nix develop --command python ...`.
- Do not commit tests; create temporary checks under `/tmp` and remove them before completion.
- Keep every source file below 300 lines and every folder at no more than 16 files.
- Stage 1 and Stage 3 training F0 extraction uses the exact shared adversarial segment.
- N and encoder KL remain full-utterance objectives.
- Stage 2 and all validation paths remain unchanged and full-utterance.

---

### Task 1: Segment-scoped F0 target interface

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Test: `/tmp/beetle_cropped_f0_checks/model_contract.py`

**Interfaces:**
- Produces: `Stage1Models.f0_target(mel: Tensor, frame_mask: Tensor) -> Tensor`.
- Produces: `Stage1Models.n_target(mel: Tensor, frame_mask: Tensor) -> Tensor`.
- Produces: `Stage1Models.segment_f0_target(mel: Tensor, frame_mask: Tensor, segment: AlignedSegments) -> Tensor`.
- Preserves: `Stage1Models.acoustic_targets(...)` for full-utterance validation.

- [ ] **Step 1: Write the failing model contract**

Create a temporary script that constructs `Stage1Models` with a recording F0
module, selects different aligned starts for two examples, and asserts:

```python
target = models.segment_f0_target(mel, mask, segment)
assert torch.equal(recorder.mel, segment.frames(mel))
assert torch.equal(recorder.mask, segment.frames(mask))
assert target.shape == (2, 32)
assert models.n_target(mel, mask).shape == (2, 40)
```

- [ ] **Step 2: Run RED**

Run: `nix develop --command python /tmp/beetle_cropped_f0_checks/model_contract.py`

Expected: fail because `segment_f0_target` and `n_target` do not exist.

- [ ] **Step 3: Add the minimal target methods**

Implement the methods without changing the existing target datatype:

```python
def f0_target(self, mel: Tensor, frame_mask: Tensor) -> Tensor:
    return self.f0_extractor(mel, frame_mask)

def n_target(self, mel: Tensor, frame_mask: Tensor) -> Tensor:
    return normalized_log_mel_energy(mel, frame_mask)

def segment_f0_target(
    self, mel: Tensor, frame_mask: Tensor, segment: AlignedSegments
) -> Tensor:
    return self.f0_target(segment.frames(mel), segment.frames(frame_mask))
```

Make `acoustic_targets` compose `f0_target` and `n_target`, preserving full
validation behavior.

- [ ] **Step 4: Run GREEN**

Run: `nix develop --command python /tmp/beetle_cropped_f0_checks/model_contract.py`

Expected: exit 0 and exact 32-frame extractor inputs.

### Task 2: Cropped Stage 1 and Stage 3 F0 supervision

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Test: `/tmp/beetle_cropped_f0_checks/training_contract.py`

**Interfaces:**
- Consumes: `Stage1Models.segment_f0_target(...)` and `Stage1Models.n_target(...)`.
- Preserves: one independently selected generator segment per Stage 1 pass and one shared posterior/conditional segment per Stage 3 pass.

- [ ] **Step 1: Write the failing trainer contract**

Use `inspect.getsource` in a temporary executable check to require that both
generator paths call `segment_f0_target`, that Stage 1 selects predicted F0 with
`segment.frames`, and that Stage 3 passes the segment into its shared acoustic
loss helper. Also assert neither generator path calls full `acoustic_targets`.

- [ ] **Step 2: Run RED**

Run: `nix develop --command python /tmp/beetle_cropped_f0_checks/training_contract.py`

Expected: fail because both trainers still call full `acoustic_targets`.

- [ ] **Step 3: Change Stage 1 generator supervision**

Compute the targets before autocast with the selected segment:

```python
f0_target = self.models.segment_f0_target(mel, frame_mask, segment)
n_target = self.models.n_target(mel, frame_mask)
segment_mask = segment.frames(frame_mask)
```

Compare `segment.frames(synthesis.acoustic.f0)` with `f0_target` under
`segment_mask`. Keep the N comparison on the complete predicted and target
tensors under `frame_mask`.

- [ ] **Step 4: Change Stage 3 shared acoustic supervision**

Compute one cropped F0 target from the generator segment and one full N target.
Change `_mean_acoustic_loss` to accept the target tensor and segment. In its
pitch branch, crop posterior/conditional predictions and the mask through the
segment; in its N branch, retain complete tensors and mask.

- [ ] **Step 5: Run GREEN and the real extractor geometry check**

Run:

```bash
nix develop --command python /tmp/beetle_cropped_f0_checks/training_contract.py
nix develop --command python -c 'import torch; from runner.nodes.training.beetle.training.runtime import load_f0_extractor; model=load_f0_extractor().cuda(); value=model(torch.zeros(2,80,32,device="cuda"),torch.ones(2,1,32,dtype=torch.bool,device="cuda")); assert value.shape == (2,32)'
```

Expected: both commands exit 0.

### Task 3: Contract documentation and compute verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/README.md`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `docs/superpowers/specs/2026-07-18-beetle-adversarial-crops-design.md`
- Test: `/tmp/beetle_cropped_f0_checks/profile.py`

**Interfaces:**
- Documents: cropped training F0 targets; full N/KL and validation.
- Reports: component and whole-step FLOP change at batch 64, 808 padded frames, and a 32-frame/9,600-sample synthesis segment.

- [ ] **Step 1: Replace stale full-F0 statements**

State explicitly that Stage 1/3 F0 target extraction and F0 loss use the shared
32-frame generator segment, while prediction, N, KL, flow objectives, Stage 2,
and validation retain their documented full geometry.

- [ ] **Step 2: Run one-step FLOP counting**

Use `torch.utils.flop_counter.FlopCounterMode` in the temporary profile script
with the same batch and shapes used for the previous measurement. Count the
cropped `F0Extractor` forward and substitute it into the reconciled Stage 1
component total.

Expected: the F0 target extractor receives `[64, 80, 32]`, and its FLOPs are
substantially below the previous 3.571738739 TFLOP full-input count.

- [ ] **Step 3: Run final verification**

Run:

```bash
nix develop --command python /tmp/beetle_cropped_f0_checks/model_contract.py
nix develop --command python /tmp/beetle_cropped_f0_checks/training_contract.py
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
git diff --check
wc -l src/runner/nodes/training/beetle/models/model.py src/runner/nodes/training/beetle/training/stage1.py src/runner/nodes/training/beetle/training/stage3.py
```

Expected: checks exit 0, no diff errors, and every listed file remains below
300 lines.

- [ ] **Step 4: Remove temporary checks and commit**

Delete `/tmp/beetle_cropped_f0_checks`, verify it no longer exists, and commit
only the intended source and documentation files.
