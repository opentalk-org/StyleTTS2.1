# iSTFTNet2-MB Vocoder Training Design

## Scope

Build a standalone Python trainer for the experimental plain `ISTFTNet2MB`
generator and `WaveUNetDiscriminator`. The trainer reads the LJSpeech WAVs
stored by the backend, trains on 8192-sample crops for five epochs, validates on
16 full utterances, saves validation media every epoch, and reports metrics and
media to the backend Aim repository. The StyleTTS3 workflow node remains out of
scope.

## Data flow

The CLI accepts the backend dataset UUID and a run output directory. It lists
non-virtual dataset audio through the shared audio CRUD facade and deterministically
holds out the last 16 records for validation. Backend packed WAVs are exported to
a local run cache in bucket groups so each storage bucket is downloaded once and
only one bounded bucket payload is resident in memory at a time.

Training files are inspected once for sample rate and frame count. Files not at
22.05 kHz fail clearly, and files shorter than 8192 frames are excluded. Dataset
workers use seeked SoundFile reads to decode exactly one random 8192-frame region;
they never decode a complete training utterance. The DataLoader shuffles training
items, drops the incomplete final batch, pins memory, keeps workers alive, and
prefetches batches. Mel extraction is vectorized on the accelerator after the
waveform batch transfer.

Validation reads each of the 16 held-out utterances in full and evaluates it
sequentially to bound accelerator memory. Waveforms are padded to the 256-sample
hop for synthesis and cropped back to their original length for exported audio.

## Audio and mel geometry

All audio remains at native LJSpeech 22.05 kHz. Generator conditioning uses 80
log-mel bins with FFT 1024, Hann window 1024, hop 256, `fmin=0`, and `fmax=8000`.
HiFi-GAN-style reflection padding with a non-centered STFT makes an 8192-sample
crop produce 32 frames, matching the generator's exact 256x output ratio.

The reconstruction objective averages log-mel L1 losses at three resolutions:

| FFT | Window | Hop | Mel bins |
| ---: | ---: | ---: | ---: |
| 512 | 512 | 128 | 80 |
| 1024 | 1024 | 256 | 80 |
| 2048 | 2048 | 512 | 80 |

## Optimization

The generator is the plain `ISTFTNet2MB`, trained from scratch. The discriminator
is `WaveUNetDiscriminator`. Each batch performs one discriminator and one generator
update using Adam with learning rate `2e-4` and betas `(0.5, 0.9)`. CUDA execution
uses bfloat16 autocast, fused Adam, cuDNN benchmarking, pinned batches, persistent
workers, and nonblocking transfers.

The discriminator uses the existing least-squares real/fake loss. The generator
objective is:

`generator_adversarial + 2 * feature_matching + 45 * mean_multi_resolution_mel`

Real discriminator features are detached for feature matching, and discriminator
parameters are frozen during the generator update.

## Validation, persistence, and Aim

Every epoch reports mean training and validation losses, learning rates, examples
per second, and audio samples per second. Validation includes multi-resolution mel
L1, waveform L1, adversarial, feature-matching, and discriminator losses.

For every validation utterance and epoch, the trainer writes:

`<output>/epoch_00001/val_00/{gt.wav,pred.wav,mel.png}`

The mel image contains ground-truth and predicted log-mels with shared color limits.
The same scalar metrics, all 16 audio pairs, and all 16 mel images are logged to an
Aim run in `AIM_REPO`. Aim initialization failures are errors because logging is a
required output, not an optional convenience.

Each epoch writes a resumable checkpoint containing generator, discriminator,
optimizers, epoch, and global step. Completion also writes a generator-only final
checkpoint.

## CLI and verification

The entry point is a directly runnable Python module beside the experimental model
files. Required options identify the dataset and output directory; batching and
worker controls are explicit. Normal execution uses five epochs and 16 validation
utterances. Smoke mode limits the training subset, validation subset, epoch count,
and optimizer steps while exercising the same data, model, loss, checkpoint, media,
and Aim paths.

Temporary tests cover crop-only reads, mel/output alignment, multi-resolution loss,
optimizer loss flow, full-audio exports, and checkpoint contents. They are removed
before completion. The real smoke run is launched through Nix. Parameter counts are
checked only after behavioral verification.
