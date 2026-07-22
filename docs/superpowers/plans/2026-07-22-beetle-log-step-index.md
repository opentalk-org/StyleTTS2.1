# Beetle Log-Step Index Implementation Plan

> **For agentic workers:** Execute inline in the current checkout. Do not create a worktree, commit, or restart training.

**Goal:** Align latent-flow training and inference by deriving every positive dyadic step size from a per-token logarithmic index and using the smallest supported step as the analytic flow anchor.

**Architecture:** `FlowTrainingSample` records a per-token integer `step_index`; sampling derives `step = 2 ** -step_index` and samples `time` on that step's valid grid. The largest index is the analytic target, smaller indices are EMA shortcut targets, and inference continues to pass the same derived physical step size.

**Tech Stack:** Python, PyTorch, Pydantic configuration, Nix development shell.

## Global Constraints

- Preserve per-token step and time sampling.
- Preserve `base_case_probability` as the probability of selecting the analytic anchor.
- Do not restart the active training process.
- Do not commit or create a worktree.
- Use temporary tests and remove them before finishing.

---

### Task 1: Indexed training samples

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/latent_flow/model.py`
- Temporary test: `src/runner/nodes/training/beetle/models/modules/latent_flow/test_log_step_tmp.py`

- [ ] Write a failing test asserting `k in [0, log2(M)]`, `d=2^-k`, grid-aligned `t`, `t+d<=1`, and no valid `d=0`.
- [ ] Run it through `nix develop --command python -m pytest ...` and confirm the old sampler fails.
- [ ] Add `step_index` to `FlowTrainingSample` and derive both `step` and valid-grid `time` from it.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Anchor and bootstrap routing

**Files:**
- Modify: `src/runner/nodes/training/beetle/losses/conditional.py`
- Modify: `src/runner/nodes/training/beetle/losses/flow.py`
- Temporary test: `src/runner/nodes/training/beetle/models/modules/latent_flow/test_log_step_tmp.py`

- [ ] Add a failing test showing the smallest shortcut teacher query uses `1/M`, never zero.
- [ ] Route analytic loss by `step_index == log2(M)` and shortcut loss by lower indices.
- [ ] Remove the zero-step substitution from the EMA teacher query.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Document and verify the contract

**Files:**
- Modify: `src/runner/nodes/training/beetle/papers/latent-flow.md`
- Remove: `src/runner/nodes/training/beetle/models/modules/latent_flow/test_log_step_tmp.py`

- [ ] Replace the theoretical `d=0` convention with the finite indexed convention used by the released Shortcut Models code.
- [ ] Run focused static compilation and relevant available tests through Nix.
- [ ] Remove the temporary test and inspect the final diff without touching unrelated changes.
