# Beetle Stage 1 Context Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this work.

**Goal:** Train Stage 1 on fixed 0.8-second contextual windows with complete source coverage and stable per-GPU shapes while leaving Stages 2 and 3 unchanged.

**Architecture:** A Stage 1-only hierarchical planner expands complete source segments lazily into fixed window descriptors. A dedicated source/collator decodes each distinct source once, creates 196-frame posterior inputs and coherent 64-frame/19,200-sample targets, and the Stage 1 model path slices the central 32 posterior frames before synthesis.

**Tech Stack:** Python, PyTorch, Pydantic, Accelerate, PostgreSQL shared audio CRUD, Nix development shell

## Global Constraints

- Run Python only through `nix develop --command python ...`.
- Work in the current checkout; do not create a branch, worktree, or subagent.
- Keep temporary verification files outside the repository and remove them before completion.
- Do not commit tests, generated audio, checkpoints, or run outputs.
- Stage 1 uses fixed `[B,192,32]` posterior and `[B,1,19200]` waveform targets.
- Stage 2 remains full-utterance and frozen under `torch.no_grad()`.
- Stage 3 code and `adversarial.segment_samples` behavior remain unchanged.

---

### Task 1: Stage 1 window planning and collation

**Files:**
- Create: `src/runner/nodes/training/beetle/data/stage1_records.py`
- Create: `src/runner/nodes/training/beetle/data/stage1_sampling.py`
- Create: `src/runner/nodes/training/beetle/data/stage1_loader.py`
- Modify: `src/runner/nodes/training/beetle/data/index.py`
- Modify: `src/runner/nodes/training/beetle/data/prefetch.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/data/__init__.py`
- Create temporarily: `/tmp/check_beetle_stage1_windows.py`

**Interfaces:**
- Produces: `Stage1WindowPlan(key, latent_start, window_index)`.
- Produces: `Stage1PlannedBatch(windows)` with exactly the configured per-GPU count.
- Produces: `Stage1Batch` containing `encoder_mel`, `encoder_mask`, `target_mel`, `frame_mask`, `waveform`, and window identities.
- Produces: `Stage1WindowPlanner.next_batch()`, `state_dict()`, and `load_state_dict()` compatible with prefetch checkpointing.

- [ ] Write a temporary failing check that constructs synthetic indexed segments, expects sequential and end-aligned 32-frame windows, expects two ranks to receive disjoint fixed-size batches, and expects state restore to reproduce the next batch exactly.
- [ ] Run it through Nix and confirm the Stage 1 window types are absent.
- [ ] Add source sample rate to `IndexedSegment`, derived by direct `audio_metadata["sample_rate"]` access, and include it in the data fingerprint.
- [ ] Implement lazy window geometry from exact source-frame, resampling, mel-hop, and posterior-rate calculations. Generate regular starts plus one end-aligned final start.
- [ ] Implement deterministic per-rank pending queues over a continuous source permutation; consume exactly `batch_size` windows per rank and checkpoint every pending descriptor.
- [ ] Add a failing collation check with two windows from one fetched clip. Assert one preprocessing call and exact `[B,80,196]`, `[B,80,64]`, `[B,1,19200]`, and Boolean-mask geometry.
- [ ] Implement the Stage 1 bulk source and collator. Decode each distinct complete source once, verify planned versus decoded frame geometry, and extract 66-frame left/right posterior context.
- [ ] Generalize the existing bounded prefetcher through typed planner/loader protocols and build the dedicated Stage 1 path without changing Stage 2/3 construction.
- [ ] Re-run the focused planner/collator check and require every assertion to pass.

### Task 2: Central posterior synthesis and Stage 1 trainer

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/loop.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/stages.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Extend temporarily: `/tmp/check_beetle_stage1_windows.py`

**Interfaces:**
- Produces: `Stage1WindowConfig(latent_frames=32)` and derived fixed geometry validation.
- Produces: `Stage1Models.reconstruct_window(encoder_mel, encoder_mask, target_mask, latent_generator, source_generator)`.
- Consumes: `Stage1Batch` directly in both Stage 1 backward passes.

- [ ] Add a failing central-posterior check comparing full-utterance and contextual mean/log-scale slices in evaluation mode.
- [ ] Add Stage 1-only explicit window configuration and require the configured compile frame count to equal the derived 196 encoder frames.
- [ ] Implement central posterior slicing at latent indices `[33:65]`, then run FeatureLinear, Decoder, and Generator against the 64-frame target mask.
- [ ] Replace Stage 1 random `AlignedSegments` selection with direct contextual-window synthesis and target losses. Keep separate deterministic discriminator/generator latent and source seeds.
- [ ] Keep Stage 1 item reporting equal to the fixed window count; generalize loop batch typing without changing Stage 2/3 item counts.
- [ ] Run the focused check and a complete discriminator plus generator backward pass; require finite gradients and exact `[B,192,32]`, `[B,512,64]`, and `[B,1,19200]` shapes.

### Task 3: Documentation and regression verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `src/runner/nodes/training/beetle/README.md`
- Verify without modifying: `src/runner/nodes/training/beetle/training/stage2_inputs.py`
- Verify without modifying: `src/runner/nodes/training/beetle/training/stage3.py`

- [ ] Document fixed contextual Stage 1 windows, full source traversal, overlap-tail policy, and unchanged later-stage behavior.
- [ ] Run the focused temporary check, package compilation, default-config load, diff hygiene, and searches proving Stage 1 no longer imports or calls `AlignedSegments` while Stage 3 still does.
- [ ] Confirm Stage 2 target/context/view posterior calls remain under `torch.no_grad()` and use full batch tensors.
- [ ] Remove `/tmp/check_beetle_stage1_windows.py`, inspect the complete diff, and commit the implementation.
