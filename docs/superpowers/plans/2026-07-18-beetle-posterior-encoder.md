# Beetle Posterior Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this work.

**Goal:** Replace Beetle's wide, long-field posterior stack with the approved 192-channel, 16-layer, dilation-one gated posterior encoder while preserving 40 Hz latent geometry.

**Architecture:** A kernel-4 stride-2 convolution establishes the 40 Hz clock before a Piper/VITS-style gated residual/skip stack. The existing posterior projection, sampling contract, masks, and downstream interfaces stay unchanged.

**Tech Stack:** Python, PyTorch, Pydantic, YAML, Nix development shell

## Global Constraints

- Run Python only through `nix develop --command python ...`.
- Work in the current checkout; do not create a branch, worktree, or subagent.
- Keep temporary verification code outside the repository and remove it before completion.
- Do not commit tests, generated artifacts, checkpoints, or run outputs.
- Preserve 80 Hz mel input and `[B,192,T/2]` 40 Hz posterior output.
- Keep the implementation local to the Beetle node family.

---

### Task 1: Gated posterior encoder

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/models/modules/audio.py`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Create temporarily: `/tmp/check_beetle_posterior_encoder.py`

**Interfaces:**
- Consumes: `PosteriorEncoderConfig` with `hidden_channels=192`, `kernel_size=5`, and `layer_count=16`.
- Produces: `AudioEncoder.forward(mel, mask, generator) -> AudioPosterior` with unchanged `[B,192,T/2]` fields.
- Produces: a gated residual/skip stack whose 16 temporal convolutions all use dilation one.

- [ ] **Step 1: Write the failing focused check**

Create `/tmp/check_beetle_posterior_encoder.py` with module-scope imports. Build
`PosteriorEncoderConfig` using `layer_count=16` and no dilation-cycle fields,
construct `AudioEncoder`, and assert:

```python
assert encoder.input_projection.out_channels == 192
assert len(encoder.stack.input_layers) == 16
assert all(layer.dilation == (1,) for layer in encoder.stack.input_layers)
assert sum(parameter.numel() for parameter in encoder.parameters()) == 7_200_960
assert 4 + 2 * (5 - 1) * len(encoder.stack.input_layers) == 132
```

Also pass an `[2,80,800]` mel tensor and mask through the encoder, assert all
four posterior tensors use the existing half-rate geometry, assert masked
suffixes are zero, and backpropagate through mean plus log scale.

- [ ] **Step 2: Run the check and verify it fails for the absent configuration**

```bash
nix develop --command env PYTHONPATH=src python /tmp/check_beetle_posterior_encoder.py
```

Expected: Pydantic rejects `layer_count` because the old posterior configuration
still requires `dilation_cycle` and `cycles`.

- [ ] **Step 3: Replace posterior configuration geometry**

In `PosteriorEncoderConfig`, replace `dilation_cycle` and `cycles` with:

```python
layer_count: int = Field(gt=0)
```

Set the default posterior configuration to `hidden_channels: 192` and
`layer_count: 16`. Preserve stride two, kernel four downsampling, kernel five,
dropout, latent width, and posterior bounds.

- [ ] **Step 4: Implement the gated residual/skip stack**

In `audio.py`, add a local `GatedResidualStack` with 16 weight-normalized
kernel-five input convolutions and matching weight-normalized one-by-one
residual/skip projections. Each layer applies tanh-sigmoid gating and dropout;
the first 15 layers update the residual stream and accumulate skip features,
while the final layer adds only its skip output. Apply the numeric mask after
each residual update and to the final result.

Construct the stack from `hidden_channels`, `kernel_size`, `layer_count`, and
`dropout`. Keep the input projection, posterior projection, sampling, and
`AudioPosterior` return contract unchanged.

- [ ] **Step 5: Run the focused check and verify it passes**

```bash
nix develop --command env PYTHONPATH=src python /tmp/check_beetle_posterior_encoder.py
```

Expected: geometry, parameter, receptive-field, masking, and gradient assertions
all pass.

- [ ] **Step 6: Update the canonical architecture description**

Update `main.md` to state that the stride-two projection precedes 16
192-channel, kernel-five, dilation-one gated residual/skip layers, with a
132-mel-frame or 1.65-second theoretical field of view.

- [ ] **Step 7: Run focused and static verification**

```bash
nix develop --command env PYTHONPATH=src python /tmp/check_beetle_posterior_encoder.py
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
rg -n "posterior\.(dilation_cycle|cycles)|dilation_cycle: \[1, 2, 4, 8, 16\]" src/runner/nodes/training/beetle
git diff --check
```

Expected: the focused check and compilation succeed, the obsolete posterior
configuration search returns no matches, and `git diff --check` prints nothing.

- [ ] **Step 8: Remove the temporary check and review the diff**

Remove `/tmp/check_beetle_posterior_encoder.py` with `apply_patch`, confirm it
does not exist, and inspect `git diff --stat` plus the complete diff before
reporting completion.
