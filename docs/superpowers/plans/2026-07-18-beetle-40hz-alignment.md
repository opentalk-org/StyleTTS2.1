# Beetle 40 Hz Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this work.

**Goal:** Keep Beetle alignment, duration targets, and alignment-expanded latent phoneme features at the StyleTTS2-native 40 Hz clock.

**Architecture:** The pretrained aligner continues to reduce hop-300 mel by two. Its raw attention is masked and converted to a hard monotonic path without interpolation; Stage 2 directly multiplies phoneme features by that hard alignment, and alignment losses consume the reduced mask without another reduction.

**Tech Stack:** Python, PyTorch, monotonic-align, Nix development shell

## Global Constraints

- Run Python only through `nix develop --command python ...`.
- Work in the current checkout; do not create a branch, worktree, or subagent.
- Keep temporary verification code outside the repository and remove it before completion.
- Do not commit tests, generated artifacts, checkpoints, or run outputs.
- Preserve 80 Hz mel/F0/N, 40 Hz posterior/latent flow, and hop-300 generator geometry.

---

### Task 1: Native-rate alignment and direct Stage 2 expansion

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/aligner.py:1-128`
- Modify: `src/runner/nodes/training/beetle/losses/alignment.py:16-72`
- Modify: `src/runner/nodes/training/beetle/losses/stage2.py:62-90,157-164`
- Modify: `src/runner/nodes/training/beetle/models/modules/conditioning.py:236-248`
- Modify: `src/runner/nodes/training/beetle/training/stage2_inputs.py:10-20,90-145,170-200`
- Modify: `src/runner/nodes/training/beetle/main.md:55-60`
- Create temporarily: `/tmp/check_beetle_40hz_alignment.py`

**Interfaces:**
- Consumes: `PhonemeAligner.forward(mel, frame_mask, phonemes, phoneme_mask)` and the configured `frame_reduction=2`.
- Produces: `AlignerOutput` whose `soft_alignment` and `hard_alignment` are `[B,P,F40]` and whose `durations` are counts of 40 Hz frames.
- Produces: `compute_alignment_losses(output, phonemes, phoneme_mask, alignment_mask, blank_id)` where `alignment_mask` is `[B,1,F40]`.
- Produces: `align_phoneme_tokens(tokens, hard_alignment) -> tuple[Tensor, Tensor]`, returning aligned features `[B,C,F40]` and Boolean mask `[B,1,F40]`.
- Produces: `Stage2LossInput.alignment_mask` equal to the posterior latent mask.

- [ ] **Step 1: Write the failing native-rate check**

Create `/tmp/check_beetle_40hz_alignment.py` with imports at module scope, a module-scope fake backbone returning four attention frames for eight mel frames, and these assertions:

```python
output = aligner(mel, frame_mask, phonemes, phoneme_mask)
assert output.soft_alignment.shape == (1, 2, 4)
assert output.hard_alignment.shape == (1, 2, 4)
assert output.durations.sum().item() == 4
```

Use phoneme IDs `[1, 2]`, token count `4`, an all-valid `[1,1,8]` mel mask, and monotonic raw attention that assigns the first two reduced frames to phoneme 1 and the last two to phoneme 2.

- [ ] **Step 2: Run the check and verify the current interpolation fails**

Run:

```bash
nix develop --command env PYTHONPATH=src python /tmp/check_beetle_40hz_alignment.py
```

Expected: `AssertionError` because the current adapter returns eight alignment frames.

- [ ] **Step 3: Keep aligner attention at reduced rate**

In `aligner.py`, remove the interpolation-only functional import and interpolation block. Build the alignment mask from the already computed reduced CTC mask:

```python
alignment_frame_mask = (~ctc_mask).unsqueeze(1)
```

Require `raw_attention.shape[2] == reduced_frames`, then form the valid matrix at 40 Hz:

```python
soft_alignment = raw_attention[:, 1 : max_phonemes + 1]
valid_matrix = phoneme_mask.unsqueeze(2) & alignment_frame_mask
```

Keep normalization, maximum path, and duration summation unchanged.

- [ ] **Step 4: Run the check and verify the alignment assertions pass**

Run the command from Step 2.

Expected: the three native-rate assertions pass.

- [ ] **Step 5: Add a failing reduced-mask loss assertion**

Extend the temporary check with the desired loss call, deliberately omitting `frame_reduction`:

```python
losses = compute_alignment_losses(
    output,
    phonemes,
    phoneme_mask,
    torch.ones(1, 1, 4, dtype=torch.bool),
    blank_id=0,
)
assert all(torch.isfinite(value) for value in (losses.s2s, losses.mono, losses.ctc))
```

Run the command from Step 2.

Expected: `TypeError` because the current loss signature still requires `frame_reduction`.

- [ ] **Step 6: Make alignment losses consume the native mask**

Rename the loss argument from `frame_mask` to `alignment_mask`, remove `frame_reduction`, validate the reduced mask against `output.soft_alignment`, and use:

```python
frame_lengths = alignment_mask.sum(dim=(1, 2))
input_lengths = frame_lengths
```

In `Stage2LossInput`, rename `frame_mask` to `alignment_mask` and remove `align_frame_reduction`. Update `compute_stage2_losses` to pass the new fields and five-argument loss signature.

- [ ] **Step 7: Run the check and verify the loss is finite**

Run the command from Step 2.

Expected: all native-rate and finite-loss assertions pass.

- [ ] **Step 8: Add a failing direct-expansion check**

Extend the temporary script without importing the desired symbol directly:

```python
conditioning = importlib.import_module(
    "runner.nodes.training.beetle.models.modules.conditioning"
)
assert hasattr(conditioning, "align_phoneme_tokens")
aligned, aligned_mask = conditioning.align_phoneme_tokens(
    torch.tensor([[[10.0, 20.0], [1.0, 2.0]]]),
    output.hard_alignment,
)
assert aligned.shape == (1, 2, 4)
assert aligned_mask.shape == (1, 1, 4)
assert aligned_mask.all()
```

Run the Task 1 command.

Expected: `AssertionError` because `align_phoneme_tokens` does not exist.

- [ ] **Step 9: Replace pairwise pooling with direct expansion**

Replace the unused `pairwise_pool_tokens` helper with `align_phoneme_tokens`. Validate rank, batch size, and phoneme length, multiply at the native alignment rate, and derive the valid-frame mask:

```python
numeric_alignment = hard_alignment.to(dtype=tokens.dtype)
aligned_mask = hard_alignment.to(dtype=torch.bool).any(dim=1, keepdim=True)
aligned = torch.bmm(tokens, numeric_alignment)
return aligned * aligned_mask.to(dtype=aligned.dtype), aligned_mask
```

- [ ] **Step 10: Run the check and verify direct expansion passes**

Run the Task 1 command.

Expected: every alignment, loss, expansion, and mask assertion passes.

- [ ] **Step 11: Use direct expansion in Stage 2**

Import `align_phoneme_tokens` instead of `pairwise_pool_tokens`. Replace the full-rate expansion and pooling with:

```python
aligned_tokens, aligned_mask = align_phoneme_tokens(
    latent_tokens,
    alignment.hard_alignment.detach(),
)
```

Keep the existing assertion that `aligned_mask.shape == posterior.mask.shape`.
Set `Stage2LossInput.alignment_mask=posterior.mask`, and remove construction of
the obsolete `align_frame_reduction` field.

- [ ] **Step 12: Update the canonical architecture document**

Replace the full-hop/pairwise paragraph in `main.md` with:

```markdown
Alignment and duration supervision remain at the aligner's native half-rate
40 Hz clock. Hard-alignment expansion maps latent phoneme features directly to
the 40 Hz posterior and LatentFlowModel timeline without interpolation or
pairwise pooling.
```

- [ ] **Step 13: Run focused and static verification**

```bash
nix develop --command env PYTHONPATH=src python /tmp/check_beetle_40hz_alignment.py
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
rg -n "pairwise_pool_tokens|align_frame_reduction|full hop-300 clock" \
  src/runner/nodes/training/beetle
git diff --check
```

Expected: the temporary check and compilation succeed; `rg` finds no obsolete
references; `git diff --check` prints nothing.

- [ ] **Step 14: Remove the temporary verification script**

Delete `/tmp/check_beetle_40hz_alignment.py` with `apply_patch`, then verify:

```bash
test ! -e /tmp/check_beetle_40hz_alignment.py
```

Expected: exit status 0.

- [ ] **Step 15: Commit the complete alignment correction**

```bash
git add src/runner/nodes/training/beetle/models/modules/aligner.py \
  src/runner/nodes/training/beetle/models/modules/conditioning.py \
  src/runner/nodes/training/beetle/losses/alignment.py \
  src/runner/nodes/training/beetle/losses/stage2.py \
  src/runner/nodes/training/beetle/training/stage2_inputs.py \
  src/runner/nodes/training/beetle/main.md
git commit -m "fix: keep beetle alignment at 40hz"
```

- [ ] **Step 16: Verify final repository state**

```bash
git status --short
git log -3 --oneline
```

Expected: clean worktree and the design plus implementation commit at the top
of history.
