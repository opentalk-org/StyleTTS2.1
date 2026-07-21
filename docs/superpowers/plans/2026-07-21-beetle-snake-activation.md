# Beetle Snake Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add StyleTTS2-compatible learned Snake activations to Beetle's generator residual paths without restarting the active training process or exceeding 4.25 GFLOPs per generated second.

**Architecture:** Define one channel-wise `SnakeActivation` module beside `ResBlock1D`. Give every dilation stage independent pre-convolution Snake modules at both convolution positions, which automatically covers the three temporal MRF branches and the harmonic-source residual path. Leave decoder normalization, frequency expansion, multiband iSTFT, PQMF, configuration, and the active process unchanged.

**Tech Stack:** Python 3.12, PyTorch, Nix development shell, temporary executable assertion tests.

## Global Constraints

- Run Python only through `nix develop --command python`.
- Keep every source file below 300 lines.
- Do not add AdaIN or AdaLN.
- Do not change harmonic count, frequency geometry, iSTFT, or PQMF.
- Keep the configured one-second profile strictly below 4.25 GFLOPs/s.
- Do not stop or restart `beetle-stage1-conditioning`.
- Remove temporary tests before handoff.

---

### Task 1: Learned periodic activations in generator residual blocks

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/convolution.py:148-194`
- Create temporarily: `tmp_tests/test_beetle_snake_activation.py`

**Interfaces:**
- Produces: `SnakeActivation(channels: int).forward(features: Tensor) -> Tensor`
- Consumes: `ResBlock1D(channels, kernel_size, dilations)` from the Beetle generator and harmonic-source adapter.

- [ ] **Step 1: Write the failing activation test**

Create the temporary executable assertion script:

```python
import torch

from runner.nodes.training.beetle.models.modules.convolution import (
    ResBlock1D,
    SnakeActivation,
)


activation = SnakeActivation(3)
features = torch.tensor(
    [[[-1.0, 0.0, 1.0], [0.25, 0.5, 0.75], [-0.5, 0.5, 1.5]]],
    requires_grad=True,
)
result = activation(features)
expected = features + torch.sin(features).square()
assert torch.allclose(result, expected)
result.square().mean().backward()
assert features.grad is not None and torch.isfinite(features.grad).all()
assert activation.alpha.grad is not None
assert torch.isfinite(activation.alpha.grad).all()

block = ResBlock1D(4, 3, (1, 3, 5))
snake_count = sum(isinstance(module, SnakeActivation) for module in block.modules())
assert snake_count == 6
block_input = torch.randn(2, 4, 20, requires_grad=True)
block_output = block(block_input)
assert block_output.shape == block_input.shape
block_output.mean().backward()
assert block_input.grad is not None and torch.isfinite(block_input.grad).all()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
nix develop --command python tmp_tests/test_beetle_snake_activation.py
```

Expected: failure importing `SnakeActivation`, because it does not exist yet.

- [ ] **Step 3: Implement the minimal Snake module and wire both activation sites**

Add beside `ResBlock1D`:

```python
class SnakeActivation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, features: Tensor) -> Tensor:
        scaled = self.alpha * features
        return features + torch.sin(scaled).square() / self.alpha
```

In `ResBlock1D.__init__`, create independent activations:

```python
self.activations1 = nn.ModuleList(
    SnakeActivation(channels) for _ in dilations
)
self.activations2 = nn.ModuleList(
    SnakeActivation(channels) for _ in dilations
)
```

Replace the forward loop with:

```python
for first, second, activation1, activation2 in zip(
    self.convs1,
    self.convs2,
    self.activations1,
    self.activations2,
    strict=True,
):
    residual = first(activation1(features))
    residual = second(activation2(residual))
    features = features + residual
```

- [ ] **Step 4: Run the test and verify GREEN**

Run:

```bash
nix develop --command python tmp_tests/test_beetle_snake_activation.py
```

Expected: exit 0 with exact initialized formula, finite gradients, six Snake modules, and unchanged block geometry.

- [ ] **Step 5: Verify generator geometry and complexity**

Run an executable Nix Python assertion that loads `default.yaml`, constructs `FeatureLinear`, `Decoder`, and `Generator`, invokes `profile_latent_audio`, and asserts:

```python
assert report.generated_samples == 24_000
assert report.generated_seconds == 1.0
assert report.gflops_per_second < 4.25
```

Expected: one second of 24 kHz output, unchanged four-band PQMF geometry, and less than 4.25 GFLOPs/s.

- [ ] **Step 6: Verify the active process was not restarted**

Run:

```bash
runuser -u user -- tmux list-panes -t beetle-stage1-conditioning \
  -F '#{pane_pid} #{pane_dead} #{pane_start_command}'
```

Expected: the original live pane PID remains active and `pane_dead` is `0`.

- [ ] **Step 7: Remove the temporary test and run static checks**

Delete `tmp_tests/test_beetle_snake_activation.py` with an explicit patch, then run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
git diff --check
```

Expected: both commands exit 0 and no temporary test remains.

- [ ] **Step 8: Commit the implementation**

```bash
git add src/runner/nodes/training/beetle/models/modules/convolution.py \
  docs/superpowers/plans/2026-07-21-beetle-snake-activation.md
git commit -m "feat: add snake generator activations"
```
