# iSTFTNet2-MB Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the standalone iSTFTNet2-MB generator, PQMF synthesis, output geometry, and optional NSF excitation to match the approved paper-first design.

**Architecture:** Keep the HiFi-GAN V2 temporal front end, reshape its concatenated MRF output into a four-bin 2D frequency space, apply three independent MB ShuffleBlocks there, and use three transposed convolutions to produce four complex subbands over 33 bins. Reconstruct fixed-length subband signals with 64-point iSTFT and a deterministic four-band PQMF synthesis bank.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, Nix development shell.

## Global Constraints

- Modify only `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py` plus temporary validation files that are removed before completion.
- Run all Python commands through `nix develop --command`.
- Do not keep committed tests in the repository.
- Do not inspect or tune parameter count until all behavioral validations pass.
- Preserve unrelated work in the dirty checkout.

---

### Task 1: Capture the architecture failures

**Files:**
- Create temporarily: `/tmp/test_istftnet2_mb.py`
- Test: `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`

**Interfaces:**
- Consumes: `ShuffleBlock2D`, `ISTFTNet2MB`, and `NSFISTFTNet2MB`.
- Produces: failing assertions for paper geometry and synthesis behavior.

- [ ] **Step 1: Write temporary tests**

Cover a 33-bin output spectrogram, exact `256T` waveform length, deterministic PQMF buffers, three separately applied low-frequency ShuffleBlocks, and a time-varying constant-F0 source.

- [ ] **Step 2: Run tests and verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_mb.py`

Expected: failures showing the current 64-bin output, shortened waveform, random PQMF, incorrect block topology, and constant NSF source.

### Task 2: Correct 2D generator geometry

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`
- Test: `/tmp/test_istftnet2_mb.py`

**Interfaces:**
- Consumes: concatenated MRF tensor shaped `[batch, 192, 4T]`.
- Produces: spectrogram tensor shaped `[batch, 8, 33, 4T]`.

- [ ] **Step 1: Implement one-operation ShuffleBlock2D**

Shuffle before splitting, transform one half through `C/2 → C → C/2`, concatenate the untouched and transformed halves, and remove the active-branch residual addition.

- [ ] **Step 2: Implement the paper frequency path**

Reshape to `[batch, 48, 4, time]`, project to 64 channels, apply three ShuffleBlocks at frequency four, and apply transposed convolutions `64 → 32 → 16 → 8` with final frequency output padding to reach 33 bins.

- [ ] **Step 3: Run focused geometry tests and verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_mb.py -k 'shuffle or spectrogram'`

Expected: selected tests pass.

### Task 3: Correct iSTFT and PQMF synthesis

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`
- Test: `/tmp/test_istftnet2_mb.py`

**Interfaces:**
- Consumes: four magnitude/phase spectrogram pairs with 33 bins and `4T` frames.
- Produces: waveform tensor `[batch, 1, 256T]`.

- [ ] **Step 1: Implement deterministic PQMF**

Design the 62-tap Kaiser-window prototype, derive 63-coefficient cosine-modulated synthesis filters, and use explicit four-way upsampling followed by synthesis filtering.

- [ ] **Step 2: Enforce iSTFT geometry**

Use FFT/window 64, hop 16, a registered Hann window, and `length=frames * 16`.

- [ ] **Step 3: Run synthesis tests and verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_mb.py -k 'pqmf or waveform'`

Expected: selected tests pass with deterministic coefficients and exact output length.

### Task 4: Correct NSF excitation

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`
- Test: `/tmp/test_istftnet2_mb.py`

**Interfaces:**
- Consumes: frame-rate F0 tensor `[batch, T]`.
- Produces: channel source tensor `[batch, source_channels, 4T]` with accumulated phase and voiced/unvoiced excitation.

- [ ] **Step 1: Implement accumulated harmonic phase**

Interpolate F0, form fundamental plus eight overtones, integrate `f / sample_rate` over time, apply voiced masking and appropriate noise, then merge harmonics through the learned linear layer and `tanh`.

- [ ] **Step 2: Run NSF test and verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_mb.py -k nsf`

Expected: constant voiced F0 yields a time-varying periodic source and unvoiced F0 remains noise-driven.

### Task 5: Verify behavior, then inspect parameters

**Files:**
- Modify if needed: `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`
- Remove: `/tmp/test_istftnet2_mb.py`

**Interfaces:**
- Consumes: completed standalone models.
- Produces: verified benchmark output and clean repository scope.

- [ ] **Step 1: Run the complete temporary suite**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_mb.py`

Expected: all tests pass.

- [ ] **Step 2: Run the standalone benchmark**

Run: `nix develop --command python src/runner/nodes/training/styletts3/testing/istftnet2_mb.py`

Expected: both models produce exactly 76,800 samples for 300 frames.

- [ ] **Step 3: Inspect parameter count for the first time**

Read the benchmark-reported counts and compare them to Table 3 without changing architecture or widths.

- [ ] **Step 4: Remove the temporary tests and check the diff**

Run: `rm /tmp/test_istftnet2_mb.py && git diff --check && git status --short`

Expected: no temporary test remains and only intended repository files are changed.
