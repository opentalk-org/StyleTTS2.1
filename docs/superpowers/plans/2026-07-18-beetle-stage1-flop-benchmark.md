# Beetle Stage 1 FLOP Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure counted FLOPs and sustained RTX 5090 utilization for the current real-data Stage 1 training step.

**Architecture:** A temporary harness loads the approved LJSpeech config, database index, latest Stage 1 checkpoint, and real batch pipeline. An eager model instance provides operation counts; a freshly restored production-compiled instance provides warmed timing without MLflow, validation, or checkpoint writes.

**Tech Stack:** Python 3.12, PyTorch `FlopCounterMode`, CUDA events, Nix.

## Global Constraints

- Benchmark Stage 1 only.
- Do not use subagents.
- Run Python only through `nix develop --command python ...`.
- Use real database batches with batch size 64 and the configured 808-frame static geometry.
- Restore model state from the existing latest Stage 1 checkpoint without changing it.
- Do not create MLflow runs, checkpoints, or permanent benchmark source.
- Remove the temporary harness after reporting.

---

### Task 1: Real-data Stage 1 benchmark harness

**Files:**
- Test: `/tmp/beetle_stage1_benchmark/run_benchmark.py`

**Interfaces:**
- Consumes: local run config, database segment index, existing checkpoint, Stage 1 trainer, and production compilation boundary.
- Produces: JSON containing workload geometry, phase FLOPs, total FLOP/step, warmed wall throughput, CUDA phase times, achieved TFLOPS, dense-BF16 utilization, and peak allocated memory.

- [ ] **Step 1: Build the read-only runtime inputs**

Load the configuration and database index with a callback whose cancellation
check is a no-op. Require Stage 1 eligibility, locate the existing latest
checkpoint, validate its configuration/data fingerprints, and build the normal
Stage 1 data pipeline with `IgnoredTokenizer` instances.

- [ ] **Step 2: Count one eager optimizer step**

Restore checkpoint model weights into an eager `Stage1Models`, build a fresh
trainer, fetch one real batch, and measure these calls independently:

```python
discriminator_flops = count(lambda: trainer.discriminator_backward(batch))
generator_flops = count(lambda: trainer.generator_backward(batch))
optimizer_flops = count(lambda: trainer.optimizer_step(0))
```

Assert batch size 64, mel frames 808, and waveform samples 242,400.

- [ ] **Step 3: Measure production-compiled throughput**

Release the eager model, restore a fresh model/trainer, apply `compile_stage1`,
and execute 10 warm-up steps. Measure 30 subsequent real optimizer steps with
one outer wall-clock interval and CUDA events around discriminator, generator,
and optimizer phases. Synchronize only after the interval.

- [ ] **Step 4: Calculate and print the report**

Use these formulas:

```python
steps_per_second = measured_steps / wall_seconds
achieved_tflops = total_flops / 1e12 * steps_per_second
bf16_utilization_percent = achieved_tflops / 209.5 * 100
```

Also report per-phase CUDA TFLOPS from counted phase FLOPs divided by mean CUDA
phase seconds. State that unsupported pointwise/FFT operations are absent from
the registered FLOP count.

- [ ] **Step 5: Verify and clean up**

Run:

```bash
nix develop --command python /tmp/beetle_stage1_benchmark/run_benchmark.py
test -z "$(git status --short)"
```

Expected: JSON report, finite positive timing/count fields, no repository
changes, and no training process. Remove `/tmp/beetle_stage1_benchmark` after
capturing the result.
