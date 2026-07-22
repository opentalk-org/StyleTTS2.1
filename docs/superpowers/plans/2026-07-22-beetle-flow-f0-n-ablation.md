# Beetle Flow F0/N Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate three controlled sample-1 WAVs that separate one-step shortcut quality, 128-step base-flow quality, and F0/N prediction quality.

**Architecture:** A temporary Beetle diagnostic loads the active run's step-8000 checkpoint, reconstructs validation sample 1 through the normal configuration and database paths, and restores model/EMA states without optimizers. It holds conditions, masks, flow noise, and waveform seed fixed while varying integration steps and decoder F0/N inputs.

**Tech Stack:** PyTorch, Beetle training/runtime modules, existing validation rendering, Nix development shell.

## Global Constraints

- Use validation sample 1 and checkpoint step 8000.
- Generate exactly the three variants in the approved design.
- Do not stop, mutate, or attach to the active trainer process.
- Run project Python only through `nix develop --command python ...`.
- Keep generated artifacts uncommitted and remove temporary diagnostic code.

---

### Task 1: Obtain the step-8000 checkpoint

**Files:**
- Inspect: `src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/checkpoints/latest.json`
- Inspect: `src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/train.log`

**Interfaces:**
- Consumes: active Beetle trainer output.
- Produces: a checkpoint whose optimizer step equals 8000.

- [ ] **Step 1: Poll progress without interrupting training**

Run `tail -3 src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/train.log`.

Expected: progress advances toward optimizer step 8000.

- [ ] **Step 2: Resolve and validate the checkpoint**

Run:

```bash
nix develop --command python -c '
from pathlib import Path
from runner.nodes.training.beetle.training.checkpoint import CheckpointManager
root = Path("src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/checkpoints")
manager = CheckpointManager(root, 1)
path = manager.latest()
assert path is not None
payload = manager.load(path)
assert payload.loop.optimizer_step == 8000, payload.loop.optimizer_step
print(path)
'
```

Expected: the checkpoint folder path and exit status 0.

### Task 2: Generate the controlled variants

**Files:**
- Create temporarily: `src/runner/nodes/training/beetle/_temporary_flow_f0_n_ablation.py`
- Generate: `src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/diagnostics/step_8000_sample_1/<variant>/`

**Interfaces:**
- Consumes: `CheckpointPayload.states`, `DefaultConditionalInputBuilder.build_validation`, `integrate_latent_flow`, and `render_validation_sample`.
- Produces: `one_step_gt_f0_n`, `steps_128_predicted_f0_n`, and `steps_128_gt_f0_n` artifact folders.

- [ ] **Step 1: Write the temporary diagnostic**

Define the variants exactly as:

```python
VARIANTS = (
    ("one_step_gt_f0_n", 1, True),
    ("steps_128_predicted_f0_n", 128, False),
    ("steps_128_gt_f0_n", 128, True),
)
```

Load the checkpoint with `CheckpointManager`, build models with the training
resource loaders, and restore model states by exact `(StateKind, name)` keys.
Load validation sample 1 through `prepare_run(...).validation`, normal
tokenizers, and `DefaultConditionalInputBuilder.build_validation`.

For each variant use:

```python
latent = integrate_latent_flow(
    ema_latent_flow,
    inputs.flow_sample.noise,
    inputs.conditions,
    inputs.latent_mask,
    steps,
)
predicted = acoustic.feature_linear(latent, inputs.latent_mask, batch.frame_mask)
decoder_acoustic = inputs.acoustic_target if use_ground_truth else predicted
decoded = acoustic.decoder(
    latent,
    decoder_acoustic.f0,
    decoder_acoustic.n,
    inputs.latent_mask,
    batch.frame_mask,
)
```

Give each generator call a fresh generator seeded by
`derive_seed(config.runtime.seed, 8000, audio_file_id, "source")`. Render the
waveform, latent, target-versus-used F0/N, mel, and alignment with
`render_validation_sample`.

- [ ] **Step 2: Compile the temporary diagnostic**

Run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/training/beetle/_temporary_flow_f0_n_ablation.py
```

Expected: exit status 0.

- [ ] **Step 3: Run the diagnostic**

Run:

```bash
nix develop --command python src/runner/nodes/training/beetle/_temporary_flow_f0_n_ablation.py
```

Expected: all three variant directories contain nonempty WAVs and plots.

### Task 3: Inspect and clean up

**Files:**
- Inspect: generated diagnostic WAVs and plots.
- Remove: `src/runner/nodes/training/beetle/_temporary_flow_f0_n_ablation.py`

**Interfaces:**
- Consumes: three rendered variant directories.
- Produces: an evidence-backed comparison with no temporary source remaining.

- [ ] **Step 1: Verify artifacts**

Run:

```bash
find src/runner/nodes/training/beetle/runs/libritts-aligned-window-8s/output/diagnostics/step_8000_sample_1 -type f -size +0c -printf '%p %s\n' | sort
```

Expected: the three variants each have the complete artifact set.

- [ ] **Step 2: Listen and inspect**

Compare each WAV and its F0, N, latent, and mel plots against the existing
step-5000 posterior and one-step artifacts. Record which intervention removes
flat F0/N and which restores intelligible audio.

- [ ] **Step 3: Remove temporary code and verify repository state**

Remove the temporary script with `apply_patch`, then run:

```bash
test ! -e src/runner/nodes/training/beetle/_temporary_flow_f0_n_ablation.py
git status --short
```

Expected: no temporary diagnostic source remains and generated artifacts stay
uncommitted.
