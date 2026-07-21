# Beetle Stage 1 Return to Clipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the active no-clipping experiment and start clipped Stage 1 training from zero.

**Architecture:** Terminate and remove only the active ablation, then use the unchanged clipped configuration with a new output directory and MLflow run.

**Tech Stack:** YAML, Python 3.12, PyTorch, MLflow

## Global Constraints

- Run project Python through `nix develop --command ...` as `user`.
- Delete `output-kl-off-no-clip` and MLflow run `1bfe5fdd02e4401ba207d32621de664d`.
- Preserve the older `output-kl-off` clipped baseline.
- Use `config-kl-off.yaml` without `--resume`.
- Launch into empty `output-kl-off-clipped-fresh`.

---

### Task 1: Remove the Active No-Clipping Run

- [ ] Confirm the active PID and its exact configuration/output paths.
- [ ] Send `SIGINT` and wait for graceful exit.
- [ ] Remove only `output-kl-off-no-clip` after validating the resolved path.
- [ ] Mark the associated MLflow run terminated and soft-delete it.

### Task 2: Launch Fresh Clipped Training

- [ ] Validate both optimizer maximum gradient norms are `10.0`.
- [ ] Confirm `output-kl-off-clipped-fresh` does not exist.
- [ ] Launch Stage 1 as `user` through Nix without `--resume`.
- [ ] Confirm a distinct MLflow run starts at step zero.

### Task 3: Verify Runtime Clipping

- [ ] Wait for step 250 diagnostics.
- [ ] Confirm all named gradient and clipping metrics exist.
- [ ] Confirm groups above norm 10 report coefficients below one and clipping flags equal to one.
- [ ] Confirm the replacement process remains healthy and the deleted run stays absent.
