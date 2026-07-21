# Beetle StyleTTS2 Conditioning Pyramid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh StyleTTS2-compatible harmonic F0 conditioning at two Beetle generator depths and transition Stage 1 reconstruction from target to predicted F0/`N` without changing PQMF geometry.

**Architecture:** Keep Beetle's existing decoder concatenation and temporal harmonic adapter. Return prepared `N`, add a second zero-initialized adapter at the four-bin frequency plane, and derive a deterministic target-to-predicted conditioning ratio from the optimizer step. Compute the harmonic STFT once and share it between both adapters.

**Tech Stack:** Python 3.12, PyTorch, Pydantic, YAML, Nix development shell

## Global Constraints

- Keep four PQMF subbands, iSTFT FFT size 60, subband hop 15, and waveform hop 300 unchanged.
- Keep complete `FeatureLinear -> Decoder -> Generator` compute below 4.25 GFLOPs per generated second.
- Do not feed `N` directly into the generator or use F0/`N` as normalization parameters.
- Keep harmonic construction under `torch.no_grad()` and retain explicit seeded randomness.
- Run project commands through `nix develop --command ...`.
- Use temporary tests only and remove them before completion.
- Work in the current checkout; do not create a worktree or branch.

---

### Task 1: Preserve Prepared N in Decoder Output

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/decoder.py`
- Create temporarily: `tmp_tests/beetle_conditioning/test_decoder_output.py`

**Interfaces:**
- Consumes: `Decoder.forward(latent, f0, n, latent_mask, frame_mask)`
- Produces: `DecoderOutput(features: Tensor, f0: Tensor, n: Tensor, mask: Tensor)`

- [x] **Step 1: Write a temporary failing decoder test**

Instantiate `Decoder(load_config().architecture.decoder)`, pass one masked batch with `L=8`, and assert `output.n.shape == (1, 16)`, padded values are zero, and training smoothing returns the prepared curve rather than the original input.

- [x] **Step 2: Run the focused test and verify failure**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_decoder_output.py`

Expected: FAIL because `DecoderOutput` has no `n` field.

- [x] **Step 3: Extend the typed output**

Change the dataclass and return construction to:

```python
@dataclass(frozen=True)
class DecoderOutput:
    features: Tensor
    f0: Tensor
    n: Tensor
    mask: Tensor

return DecoderOutput(
    features * numeric_frame_mask,
    prepared_f0 * numeric_frame_mask[:, 0],
    prepared_n * numeric_frame_mask[:, 0],
    boolean_frame_mask,
)
```

- [x] **Step 4: Run the focused test and verify pass**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_decoder_output.py`

Expected: PASS.

### Task 2: Add the Spectral Harmonic Adapter

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/generator.py`
- Create temporarily: `tmp_tests/beetle_conditioning/test_generator_pyramid.py`

**Interfaces:**
- Consumes: shared harmonic features `[B,242,5T]` and frequency-entry features `[B,64,4,5T]`
- Produces: `SpectralSourceAdapter.forward(source: Tensor) -> Tensor[B,64,4,5T]`

- [x] **Step 1: Write temporary failing generator tests**

Assert the generator exposes `spectral_source`, its output matches `[B,64,4,5T]`, its final projection starts at zero, equal source seeds produce equal waveforms, different seeds change them, and adapter parameters receive finite gradients.

- [x] **Step 2: Run tests and verify failure**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_generator_pyramid.py`

Expected: FAIL because `spectral_source` does not exist.

- [x] **Step 3: Add a focused adapter**

Add a top-level `SpectralSourceAdapter` in `generator.py`:

```python
class SpectralSourceAdapter(nn.Module):
    def __init__(self, source_channels: int, channels: int, frequency_bins: int) -> None:
        super().__init__()
        self.channels = channels
        self.frequency_bins = frequency_bins
        self.entry = nn.Conv1d(source_channels, channels * frequency_bins, 1)
        self.residual = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, source: Tensor) -> Tensor:
        batch, _, frames = source.shape
        projected = self.entry(source)
        projected = projected.view(
            batch,
            self.channels,
            self.frequency_bins,
            frames,
        )
        return self.residual(projected)
```

- [x] **Step 4: Reuse the harmonic STFT at both injection sites**

Compute `harmonic = self.harmonic_features(...)` once. Feed it to the existing temporal projection and to `self.spectral_source`. Add the spectral result immediately after `self.frequency_entry(features)` and before the frequency shuffle blocks. Pass the spectral source into `_subband_spectrogram` as a typed tensor argument.

- [x] **Step 5: Run generator tests and verify pass**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_generator_pyramid.py`

Expected: PASS with exact waveform geometry `[1,1,24000]` for 80 input frames.

### Task 3: Transition Stage 1 to Predicted Conditioning

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Create: `src/runner/nodes/training/beetle/models/parameters.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/stage1.py`
- Create temporarily: `tmp_tests/beetle_conditioning/test_conditioning_schedule.py`

**Interfaces:**
- Produces: `Stage1ConditioningConfig(predicted_start_step: int, transition_steps: int)`
- Produces: `Stage1ConditioningSchedule.predicted_ratio(optimizer_step: int) -> float`
- Consumes: target and predicted `AcousticFeatures` inside `Stage1Models.reconstruct_window`

- [x] **Step 1: Write temporary failing schedule tests**

With `predicted_start_step=50_000` and `transition_steps=50_000`, assert ratios are exactly `0.0` at steps 0 and 50,000, `0.5` at 75,000, and `1.0` at and after 100,000. Assert a start of zero uses predictions immediately and a start beyond the run keeps target conditioning throughout. Assert rendered decoder conditioning equals target, a 50/50 interpolation, and prediction at those ratios.

- [x] **Step 2: Run tests and verify failure**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_conditioning_schedule.py`

Expected: FAIL because the configuration and schedule do not exist.

- [x] **Step 3: Add required configuration**

Add a strict top-level configuration:

```python
class Stage1ConditioningConfig(StrictConfigModel):
    predicted_start_step: int = Field(ge=0)
    transition_steps: int = Field(gt=0)
```

Add `stage1_conditioning: Stage1ConditioningConfig` to `BeetleConfig` and the following required YAML:

```yaml
stage1_conditioning:
  predicted_start_step: 50000
  transition_steps: 50000
```

Set discriminator, generator-adversarial, and feature-matching schedules to `start_step: 0` with `warmup_steps: 3000`; keep this independent from the conditioning transition.

- [x] **Step 4: Implement the deterministic schedule**

Add a frozen top-level schedule type in `stage1.py`:

```python
@dataclass(frozen=True)
class Stage1ConditioningSchedule:
    predicted_start_step: int
    transition_steps: int

    def predicted_ratio(self, optimizer_step: int) -> float:
        if self.predicted_start_step == 0:
            return 1.0
        transition_position = optimizer_step - self.predicted_start_step
        bounded = min(max(transition_position, 0), self.transition_steps)
        return bounded / self.transition_steps
```

Construct it from `Stage1ConditioningConfig`, derive the ratio from the current optimizer step for both discriminator and generator synthesis, and report a `conditioning_predicted_ratio` metric.

- [x] **Step 5: Blend conditioning inside the single posterior pass**

Change `Stage1Models.reconstruct_window` to accept target acoustic features and `predicted_ratio`. After computing predicted `acoustic`, form typed decoder conditioning:

```python
target_ratio = 1.0 - predicted_ratio
decoder_acoustic = AcousticFeatures(
    f0=target_acoustic.f0 * target_ratio + acoustic.f0 * predicted_ratio,
    n=target_acoustic.n * target_ratio + acoustic.n * predicted_ratio,
)
```

Use this value for the decoder without repeating the audio encoder or harmonic source.

- [x] **Step 6: Match validation to the configured schedule**

Use the same ratio at the validation step and include `conditioning_predicted_ratio` in validation metrics. Validation artifacts continue to contain target and predicted F0/`N` pairs.

- [x] **Step 7: Run schedule and configuration tests**

Run: `nix develop --command python tmp_tests/beetle_conditioning/test_conditioning_schedule.py`

Expected: PASS.

### Task 4: Integrated Verification and Cleanup

**Files:**
- Verify: `src/runner/nodes/training/beetle/`
- Remove: `tmp_tests/beetle_conditioning/`

**Interfaces:**
- Consumes: complete configured `FeatureLinear -> Decoder -> Generator` path
- Produces: verified 24 kHz waveform and complexity report

- [x] **Step 1: Run all temporary conditioning tests together**

Run each executable assertion script in `tmp_tests/beetle_conditioning/` through `nix develop --command python`.

Expected: all tests PASS.

- [x] **Step 2: Run static checks on changed modules**

Run: `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`

Run: `nix develop --command python -m ruff check src/runner/nodes/training/beetle`

Expected: both commands exit 0.

- [x] **Step 3: Run the configured complexity profile**

Load `default.yaml`, instantiate `FeatureLinear`, `Decoder`, and `Generator`, call `profile_latent_audio`, and assert waveform geometry is one second, the report is finite, and `gflops_per_second < 4.25`.

Expected: PASS without changing PQMF subbands or output hop.

- [x] **Step 4: Remove temporary tests**

Delete `tmp_tests/beetle_conditioning/` with an explicit patch and confirm no temporary files are tracked.

- [x] **Step 5: Review the final diff**

Run: `git diff --check` and inspect only the intended Beetle configuration, model, training, validation, spec, and plan changes.

Expected: no whitespace errors, generated files, caches, weights, audio, or unrelated modifications.
