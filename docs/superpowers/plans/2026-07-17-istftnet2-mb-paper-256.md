# iSTFTNet2-MB Paper-256 Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful 22.05 kHz, hop-256 iSTFTNet2-MB generator and training profile using the requested StyleTTS GAN backend.

**Architecture:** A frozen profile model carries signal and mel geometry through the existing data and training layers. A separate paper generator reuses only architecture blocks that are identical to the native-hop-300 model, while its `C4-I16-B4` synthesis path stays explicit. The existing CLI selects the generator and profile without changing native defaults.

**Tech Stack:** Python, PyTorch, torchaudio, SoundFile, Pydantic-free frozen dataclasses/enums, argparse, MLflow, Nix development shell.

## Global Constraints

- Preserve the running training process and do not launch another CUDA training workload.
- Preserve all pre-existing uncommitted changes in the training files.
- Keep `native_300` as the CLI default with unchanged behavior.
- Implement `paper_256` at 22,050 Hz, FFT/window 1024, hop 256, and `C4-I16-B4` synthesis.
- Require the existing StyleTTS GAN backend for `paper_256`.
- Default the paper profile to 2,500,000 optimizer steps; permit `--max-steps` for bounded smoke checks.
- Run Python and pytest commands only through `nix develop --command`.
- Keep source files below 300 lines and remove temporary tests before completion.

---

### Task 1: Typed training profiles

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/profiles.py`
- Create temporarily: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: `VocoderProfile`, `MelGeometry`, `SignalGeometry`, `NATIVE_300`, `PAPER_256`, and `profile_geometry(profile)`.
- Consumes: no training modules, preventing configuration/import cycles.

- [ ] **Step 1: Write the failing profile test**

```python
from runner.nodes.training.styletts3.testing.vocoder_training.profiles import (
    PAPER_256,
    VocoderProfile,
    profile_geometry,
)


def test_paper_profile_geometry():
    profile = profile_geometry(VocoderProfile.PAPER_256)
    assert profile is PAPER_256
    assert (profile.sample_rate, profile.segment_samples, profile.synthesis_hop) == (
        22_050,
        8_192,
        256,
    )
    assert (profile.conditioning.n_fft, profile.conditioning.win_length) == (1024, 1024)
    assert (profile.conditioning.hop_length, profile.conditioning.fmin) == (256, 80.0)
    assert profile.conditioning.fmax == 7600.0
    assert profile.target_steps == 2_500_000
```

- [ ] **Step 2: Verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py`

Expected: FAIL because `vocoder_training.profiles` does not exist.

- [ ] **Step 3: Implement frozen profile values**

```python
class VocoderProfile(str, Enum):
    NATIVE_300 = "native_300"
    PAPER_256 = "paper_256"


@dataclass(frozen=True)
class MelGeometry:
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float


@dataclass(frozen=True)
class SignalGeometry:
    sample_rate: int
    segment_samples: int
    synthesis_hop: int
    conditioning: MelGeometry
    reconstruction: tuple[MelGeometry, ...]
    target_steps: int | None
```

Define native values from the current `geometry.py` and `mel.py` unchanged. Define paper conditioning as `1024/1024/256/80/80/7600`, paper reconstruction as `1024/1024/256/80/0/11025`, and paper target steps as `2_500_000`. Use direct enum indexing in `profile_geometry` so unknown profiles fail.

- [ ] **Step 4: Verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py`

Expected: PASS.

### Task 2: Exact paper generator

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/paper_istftnet2_mb.py`
- Modify temporarily: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: `PaperISTFTNet2MB`, accepting mel tensors `(batch, 80, frames)` and returning `(batch, 1, frames * 256)`.
- Consumes: `PQMF` from the shared iSTFTNet2-MB package; paper-specific neural blocks remain local.

- [ ] **Step 1: Add failing architecture tests**

```python
def test_paper_generator_geometry():
    model = PaperISTFTNet2MB().eval()
    mel = torch.randn(2, 80, 4)
    with torch.no_grad():
        temporal = model.temporal_features(mel)
        spectrogram = model.subband_spectrogram(temporal)
        waveform = model(mel)
    assert temporal.shape == (2, 64, 16)
    assert spectrogram.shape == (2, 8, 33, 16)
    assert waveform.shape == (2, 1, 1024)
```

Also assert the constructor rejects non-paper channel, band, and FFT arguments.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k generator`

Expected: FAIL because `paper_istftnet2_mb` does not exist.

- [ ] **Step 3: Implement `C4-I16-B4`**

Use weight-normalized `Conv1d(80, 128, 7, padding=3)`, then `ConvTranspose1d(128, 64, 8, stride=4, padding=2)`. Implement paper-local weight-normalized residual and ShuffleBlocks so the reference does not inherit the unnormalized native blocks. Apply concatenated 1D MRF branches, reshape the 192 channels to `(48, 4, time)`, and project to 64 2D channels. Apply three modified ShuffleBlocks. Use frequency transposed convolutions `(64->32, kernel=(4,3), padding=(1,1))`, `(32->16, kernel=(4,3), padding=(1,1))`, and `(16->8, kernel=(3,3), padding=(0,1))` so the frequency axis is exactly `4->8->16->33`.

For each subband, convert predicted log magnitude and bounded phase to a complex spectrum, then use 64-point Hann iSTFT at hop/window `16/64`. Reconstruct the four bands with the shared PQMF and assert the final 256x shape.

- [ ] **Step 4: Verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k generator`

Expected: PASS with exact intermediate and output shapes.

### Task 3: Profile-driven audio and mel processing

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/vocoder_training/audio_data.py`
- Modify: `src/runner/nodes/training/styletts3/testing/vocoder_training/mel.py`
- Modify temporarily: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: profile-aware `prepare_backend_audio`, `build_train_loader`, `LogMelSpectrogram`, `MultiResolutionMelLoss`, `conditioning_mel`, and `pad_to_hop`.
- Consumes: `SignalGeometry`; no module-level signal fallback is introduced.

- [ ] **Step 1: Add failing cache, segment, and mel tests**

Write a temporary 24 kHz WAV and assert `_write_cached_wav(..., PAPER_256)` produces 22.05 kHz audio at a cache path containing `22050`. Assert `LogMelSpectrogram(PAPER_256.conditioning, PAPER_256.sample_rate)` maps an 8,192-sample batch to 32 frames and `pad_to_hop(waveform, PAPER_256.synthesis_hop)` pads to a multiple of 256. Retain equivalent assertions for `NATIVE_300`.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k 'cache or mel or padding'`

Expected: FAIL because the current functions use native module globals.

- [ ] **Step 3: Thread geometry through data and mel modules**

Store `segment_samples` on `StreamingCropDataset`; use it in length calculation and reads. Pass `SignalGeometry` to audio inspection, cache naming, resampling, loader construction, and mel factories. Build mel filterbanks from each `MelGeometry` and the selected sample rate. Keep native constants only in `profiles.py`; every caller passes its selected profile explicitly.

- [ ] **Step 4: Verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k 'cache or mel or padding'`

Expected: PASS for both profiles.

### Task 4: Step-bounded shared trainer

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/vocoder_training/trainer.py`
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/training_config.py`
- Modify temporarily: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: `TrainingConfig.max_steps`, profile-aware `train_batch`, `validate_epoch`, and `train_vocoder` accepting any `nn.Module` generator plus `SignalGeometry`.
- Consumes: the existing optimizers, reporter, metrics, and discriminator backend without changing their active-job semantics.

- [ ] **Step 1: Add a failing step-bound test**

Use tiny recording generator/discriminator modules and a two-batch loader. Configure multiple epochs with `max_steps=3`; assert exactly three optimizer batches run and the final generator artifact is written. Assert native configuration with `max_steps=None` retains `epochs * effective_steps_per_epoch` behavior.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k step_bound`

Expected: FAIL because `TrainingConfig` has no global step bound or signal profile.

- [ ] **Step 3: Generalize without changing loss flow**

Move `TrainingConfig` into `training_config.py` so the touched trainer returns below the 300-line repository limit. Change generator annotations to `nn.Module`. Add `max_steps: int | None` to `TrainingConfig` and pass `SignalGeometry` into training. Construct mel modules from that geometry, pad validation with its synthesis hop, and compute the total step target as `max_steps` when set or `effective_steps_per_epoch * epochs` otherwise. Repeat loader epochs until the target is reached and break immediately after the target step. Keep current StyleTTS discriminator, relative losses, metric names, and optimizer ordering intact.

- [ ] **Step 4: Verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k step_bound`

Expected: PASS.

### Task 5: CLI profile selection and MLflow metadata

**Files:**
- Modify: `src/runner/nodes/training/styletts3/testing/train_istftnet2_mb.py`
- Modify temporarily: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: `--profile {native_300,paper_256}`, `--max-steps`, `build_generator(profile)`, and strict discriminator resolution.
- Consumes: `PaperISTFTNet2MB`, `ISTFTNet2MB`, profile geometry, existing reporter, and existing MLflow helper.

- [ ] **Step 1: Add failing CLI tests**

```python
def test_cli_defaults_to_native_profile():
    args = parse_args(REQUIRED_ARGS)
    assert args.profile == VocoderProfile.NATIVE_300.value


def test_paper_profile_requires_styletts():
    with pytest.raises(ValueError, match="paper_256 requires the styletts discriminator"):
        resolve_discriminator(VocoderProfile.PAPER_256, "wave_unet")
```

Assert `build_generator(PAPER_256)` returns `PaperISTFTNet2MB` and the paper default resolves to StyleTTS when `--discriminator` is omitted.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k cli`

Expected: FAIL because the CLI has no profile or global step option.

- [ ] **Step 3: Compose the selected profile**

Add enum-derived argparse choices. Make the discriminator argument default `None`, resolving native to its current Wave-U-Net default and paper to StyleTTS. Reject explicit Wave-U-Net with paper. Pass signal geometry into audio, reporter, and trainer construction. Record profile, full signal/mel geometry, target/max steps, and `paper_deviation="styletts_gan_with_relative_loss"` in MLflow. Keep the experiment name distinct for the paper profile.

- [ ] **Step 4: Verify GREEN**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k cli`

Expected: PASS.

### Task 6: Integration verification and cleanup

**Files:**
- Verify: all modified source files
- Remove: `/tmp/test_istftnet2_paper_profile.py`

**Interfaces:**
- Produces: a ready-to-run paper command without starting a second CUDA job.
- Consumes: the completed generator, profile, StyleTTS backend, and CLI.

- [ ] **Step 1: Run the full temporary suite**

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py`

Expected: all profile, generator, data, trainer, and CLI tests PASS.

- [ ] **Step 2: Run CPU integration**

Add this integration test to the temporary suite, then run it through Nix:

```python
def test_paper_generator_and_styletts_gan_backward():
    generator = PaperISTFTNet2MB()
    discriminator = StyleTTSBackend()
    mel = torch.randn(1, 80, 8)
    real = torch.randn(1, 1, 2048)
    fake = generator(mel)
    real_eval, fake_eval = discriminator.evaluate_pair(real, fake)
    loss = (
        discriminator.generator_adv_loss(real_eval, fake_eval)
        + 2.0 * discriminator.feature_matching_loss(real_eval, fake_eval)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert not torch.cuda.is_initialized()
```

Run: `nix develop --command pytest -q /tmp/test_istftnet2_paper_profile.py -k styletts_gan_backward`

Expected: PASS with finite loss and populated generator gradients.

- [ ] **Step 3: Verify native behavior and static quality**

Run `nix develop --command python -m compileall -q src/runner/nodes/training/styletts3`, `git diff --check`, `wc -l` over every touched source file, and `nix develop --command python -m runner.nodes.training.styletts3.testing.train_istftnet2_mb --help`. Expected: all commands succeed, no touched source file exceeds 300 lines, and help lists both profiles.

- [ ] **Step 4: Confirm the active job is untouched**

Read the original training PID from `ps` and confirm it is still the same process or has exited naturally. Do not signal or restart it. Confirm no new paper training process was launched.

- [ ] **Step 5: Remove temporary tests and re-run final checks**

Remove `/tmp/test_istftnet2_paper_profile.py` with `apply_patch`, then repeat compile, CLI help, `git diff --check`, and `git status --short`. Expected: no temporary test remains and only intended source/docs changes plus the user's pre-existing changes are present.
