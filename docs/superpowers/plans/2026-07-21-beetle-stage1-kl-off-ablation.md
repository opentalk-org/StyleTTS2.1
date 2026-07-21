# Beetle Stage 1 KL-Off Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the active KL-on baseline and start a clean LJSpeech Stage 1 run with encoder KL weight zero.

**Architecture:** The baseline output remains immutable. A copied run configuration changes only the scheduled encoder KL weight, and a new standalone process writes to a distinct empty output directory and creates a distinct MLflow run.

**Tech Stack:** Nix development shell, Python Beetle Stage 1 CLI, PyTorch, MLflow

## Global Constraints

- Run all project commands through `nix develop --command ...`.
- Run training as `user`, not root.
- Keep the current KL-on output and MLflow run unchanged.
- Do not pass `--resume`; the ablation starts at optimizer step 0.
- Keep evaluation every 4,000 optimizer steps and batch size 64 with accumulation 1.

---

### Task 1: Record the Mel-Loss Comparison

**Files:**
- Read: `src/runner/nodes/training/beetle/losses/acoustic.py`
- Read: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml`
- External: `https://github.com/yl4579/StyleTTS2/blob/main/losses.py`
- External: `https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml`
- External: `https://github.com/yl4579/StyleTTS2/blob/main/train_first.py`

**Interfaces:**
- Consumes: the original StyleTTS2 loss implementation and the active Beetle configuration.
- Produces: an evidence-backed comparison of transforms, reductions, schedules, and effective weights.

- [ ] **Step 1: Verify both implementations and configuration values**

Run:

```bash
nix develop --command rg -n "MultiResolution|SpectralResolution|reconstruction:|encoder_kl:" src/runner/nodes/training/beetle/losses/acoustic.py src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml
```

Expected: three Beetle resolutions, reconstruction weight `5.0`, and encoder KL weight `1.0` before the ablation.

- [ ] **Step 2: Report the comparison with direct source links**

Expected: distinguish StyleTTS2's pre-TMA unweighted mel-only phase from its post-TMA `lambda_mel: 5` phase, and distinguish that schedule from Beetle's always-weighted reconstruction plus adversarial warmup.

### Task 2: Stop the KL-On Baseline Cleanly

**Files:**
- Preserve: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-compiled/`

**Interfaces:**
- Consumes: active Stage 1 PID and its installed SIGTERM cancellation callback.
- Produces: a stopped baseline process whose output remains intact.

- [ ] **Step 1: Send SIGTERM to the exact active PID**

Run:

```bash
kill -TERM 637163
```

Expected: the process finishes its exact-resume boundary and exits without deleting output.

- [ ] **Step 2: Verify termination and preserved output**

Run:

```bash
ps -p 637163
du -sh src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-compiled
```

Expected: `ps` has no process row and the baseline directory remains populated.

### Task 3: Create and Launch the KL-Off Run

**Files:**
- Create: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml`
- Create at runtime: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off/`

**Interfaces:**
- Consumes: the baseline configuration.
- Produces: a standalone fresh Stage 1 training process with effective encoder KL weight `0.0`.

- [ ] **Step 1: Copy the baseline configuration mechanically**

Run:

```bash
nix develop --command cp src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml
```

Expected: the copied YAML exists and initially matches the baseline byte-for-byte.

- [ ] **Step 2: Change only the encoder KL scheduled weight**

Apply this exact YAML change to `config-kl-off.yaml`:

```yaml
encoder_kl: {value: 0.0, start_step: 0, warmup_steps: 0}
```

Expected: a diff against `config.yaml` contains only `1.0` to `0.0` on the encoder KL line.

- [ ] **Step 3: Launch as the non-root training user without resume**

Run:

```bash
nix develop --command setsid -f runuser -u user -- env HOME=/home/user TMPDIR=/tmp XDG_CACHE_HOME=/home/user/.cache XDG_DATA_HOME=/home/user/.local/share XDG_STATE_HOME=/home/user/.local/state RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow MLFLOW_TRACKING_URI=http://127.0.0.1:7860 MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=runflow AWS_SECRET_ACCESS_KEY=runflow-secret AWS_DEFAULT_REGION=us-east-1 bash -lc 'exec python -m runner.nodes.training.beetle.scripts.train_stage1 --config src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml --output src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off > src/runner/nodes/training/beetle/runs/ljspeech-stage1/train-kl-off.log 2>&1'
```

Expected: one new Python process owns the GPU and references the KL-off paths; no `--resume` argument is present.

### Task 4: Verify the Fresh Run

**Files:**
- Inspect: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off/`

**Interfaces:**
- Consumes: the KL-off process, its log, and MLflow tracking state.
- Produces: proof that the run began from zero and completed at least one optimizer step.

- [ ] **Step 1: Verify configuration invariants**

Run:

```bash
nix develop --command diff -u src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml
```

Expected: only the encoder KL weight differs; batch size is 64, accumulation is 1, and validation interval is 4,000.

- [ ] **Step 2: Verify process ownership and progress**

Run:

```bash
nix develop --command pgrep -af "train_stage1.*config-kl-off.yaml"
nix develop --command tail -n 80 src/runner/nodes/training/beetle/runs/ljspeech-stage1/train-kl-off.log
```

Expected: the process is owned by `user`, reports a new MLflow run, begins at step 0, and completes at least optimizer step 1 without a traceback.
