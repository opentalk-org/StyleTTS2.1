# Beetle StyleTTS2 Harmonic Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Beetle's harmonic excitation use StyleTTS2's float32 frame-rate phase construction without propagating waveform gradients through oscillator phase into F0.

**Architecture:** Keep `HarmonicSource` as Beetle's local learned harmonic merger, but move stochastic oscillator construction into a detached float32 operation. Accumulate phase at frame rate and linearly interpolate it to sample rate, preserving Beetle's explicit seeded generator and the existing downstream harmonic STFT and generator geometry.

**Tech Stack:** Python, PyTorch, pytest, Nix development shell

## Global Constraints

- Keep the implementation inside `src/runner/nodes/training/beetle/`; do not import implementation modules from another node family.
- Preserve Beetle's seeded randomness, masks, configured harmonic count, and waveform geometry.
- Keep the harmonic merge and all downstream generator modules trainable.
- Run Python and pytest only through `nix develop --command`.
- Use temporary tests and remove them before completion.

---

### Task 1: Correct harmonic excitation precision, phase, and gradient boundary

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/vocoder.py:9-52`
- Test temporarily: `/tmp/test_beetle_harmonic_source.py`

**Interfaces:**
- Consumes: `HarmonicSource.forward(f0: Tensor, generator: torch.Generator) -> Tensor`
- Produces: the same `[B, 1, frames * output_hop]` waveform interface in float32, with gradients for `HarmonicSource.merge` but no gradient dependency on `f0`

- [ ] **Step 1: Write failing behavior tests**

Create `/tmp/test_beetle_harmonic_source.py` with tests that instantiate a short-hop `HarmonicSource`, pass bfloat16 F0 with `requires_grad=True`, and assert:

```python
def test_harmonic_source_uses_float32_without_f0_gradient():
    source = HarmonicSource(24_000, 16, 8)
    f0 = torch.full((1, 16), 180.0, dtype=torch.bfloat16, requires_grad=True)
    output = source(f0, torch.Generator().manual_seed(7))
    assert output.dtype == torch.float32
    assert torch.autograd.grad(output.square().mean(), f0, allow_unused=True)[0] is None


def test_harmonic_source_keeps_merge_trainable_and_seeded():
    source = HarmonicSource(24_000, 16, 8)
    f0 = torch.full((1, 16), 180.0, dtype=torch.bfloat16)
    first = source(f0, torch.Generator().manual_seed(7))
    repeated = source(f0, torch.Generator().manual_seed(7))
    different = source(f0, torch.Generator().manual_seed(8))
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
    torch.autograd.grad(first.square().mean(), tuple(source.merge.parameters()))
```

Add a constant-F0 check against a float32 analytical oscillator and an output-shape check for `[1, 1, 256]`.

- [ ] **Step 2: Run the temporary tests and verify RED**

Run:

```bash
nix develop --command python -m pytest -q /tmp/test_beetle_harmonic_source.py
```

Expected: failures because the current output follows bfloat16 F0, retains an F0 gradient, and uses sample-rate cumulative phase.

- [ ] **Step 3: Implement the minimal StyleTTS2-compatible source**

In `HarmonicSource`, construct float32 sampled F0 and oscillator values inside a detached block. Generate seeded overtone phases, reduce phase increments to frame rate, accumulate phase there, multiply by `output_hop`, and linearly interpolate phase back to sample rate. Apply the voiced mask and seeded noise in float32. Pass the detached harmonic tensor through the existing learned `merge` and `tanh` outside the detached block.

- [ ] **Step 4: Run the temporary tests and verify GREEN**

Run:

```bash
nix develop --command python -m pytest -q /tmp/test_beetle_harmonic_source.py
```

Expected: all harmonic-source tests pass.

- [ ] **Step 5: Verify the complete generator contract**

Run a Nix Python check using `load_config`, `Generator`, 80 input frames, bfloat16 autocast, and seeded generation. Require output shape `[1, 1, 24000]`, finite samples, no F0 gradient through `harmonic_features`, and gradients on the learned harmonic merge.

- [ ] **Step 6: Remove temporary tests and run final checks**

Remove `/tmp/test_beetle_harmonic_source.py`, then run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/training/beetle/models/modules/vocoder.py src/runner/nodes/training/beetle/models/modules/generator.py
git diff --check
wc -l src/runner/nodes/training/beetle/models/modules/vocoder.py
```

Expected: compilation and diff checks succeed, and `vocoder.py` remains below 300 lines.

- [ ] **Step 7: Commit the implementation**

```bash
git add src/runner/nodes/training/beetle/models/modules/vocoder.py
git commit -m "fix: match Beetle harmonic source to StyleTTS2"
```
