# Beetle StyleTTS2 Conditioning Pyramid

## Goal

Make Beetle's F0 and `N` conditioning more faithful to StyleTTS2 while
preserving Beetle's fast iSTFTNet2-MB generator, four-band PQMF synthesis,
300-sample output hop, deterministic source generation, and inference compute
budget.

The change should use StyleTTS2 as a structural baseline. It should not copy
StyleTTS2's expensive full-rate temporal generator or introduce conditioning
mechanisms that StyleTTS2 does not use.

## StyleTTS2 Reference Behavior

StyleTTS2 has two distinct conditioning paths.

The decoder receives full-rate F0 and `N` curves. During training it randomly
smooths them, projects each curve from `2L` to `L` with a stride-two
convolution, and concatenates both projections with the content features. It
repeats the projected F0 and `N`, together with a content residual, at every
decoder block. The final decoder block restores the `2L` frame rate.

F0 and `N` do not parameterize AdaLN or AdaIN. They are ordinary feature
channels concatenated into the decoder. StyleTTS2's decoder blocks themselves
use style-conditioned AdaIN, and its prosody predictor uses style-conditioned
AdaIN blocks when predicting F0 and `N`; those are separate mechanisms.

The generator receives the prepared full-rate F0 but does not receive `N`
directly. It converts F0 into a stochastic harmonic excitation, transforms the
excitation into magnitude-and-phase features, and processes a separate source
branch at each generator upsampling stage before adding it to the main path.
`N` affects the waveform only through the decoded acoustic features.

## Current Beetle Behavior

Beetle's decoder already preserves the important StyleTTS2 topology:

- full-rate F0 and `N` are independently smoothed during training;
- separate stride-two convolutions project both curves to latent rate;
- both projections enter the encoder and all four decoder blocks;
- a projected latent residual accompanies the repeated conditioning;
- the final block restores full frame rate; and
- prepared full-rate F0 is returned to the generator.

Beetle intentionally replaces style-conditioned AdaIN with masked,
style-independent residual blocks. It also masks smoothing, projections,
normalization, and padded output regions.

The larger departure is in the generator. Beetle computes a StyleTTS2-like
harmonic excitation and STFT, but projects and adds the source only once before
the temporal MRF. StyleTTS2 refreshes the learned harmonic source at multiple
generator depths.

## Chosen Architecture

Keep the decoder topology and add a two-level harmonic conditioning pyramid to
the generator. The harmonic excitation and its STFT are computed once and
shared by both source adapters.

```text
decoder features [B,512,T]
    -> input projection [B,128,T]
    -> temporal upsample x5 [B,64,5T]
    +  source adapter A [B,64,5T]
    -> temporal MRF [B,192,5T]
    -> reshape [B,48,4,5T]
    -> frequency entry [B,64,4,5T]
    +  source adapter B [B,64,4,5T]
    -> frequency shuffles
    -> frequency upsampling 4 -> 8 -> 16 -> 31 bins
    -> four subband complex spectrograms
    -> multiband iSTFT
    -> PQMF synthesis
    -> waveform [B,1,300T]
```

This creates two source additions separated by the temporal MRF and the
temporal-to-frequency transition. It is the closest useful analogue to
StyleTTS2's repeated source branches within Beetle's one-stage temporal,
multistage frequency generator.

### Source adapter A

Retain the current temporal source path:

```text
harmonic STFT [B,242,5T]
    -> Conv1d 242 -> 64, kernel 1
    -> ResBlock1D, kernel 11, dilations 1/3/5
    -> add to the temporally upsampled decoder features
```

The addition occurs before the temporal MRF. Existing masks continue to be
applied at the five-times frame rate.

### Source adapter B

Add a lightweight spectral source path from the same harmonic STFT:

```text
harmonic STFT [B,242,5T]
    -> Conv1d 242 -> 256, kernel 1
    -> reshape [B,64,4,5T]
    -> lightweight 2D residual adapter
    -> add after the generator's frequency-entry convolution
```

The four frequency bins align with the generator's initial frequency plane.
The adapter therefore refreshes F0-derived information after the temporal MRF
without introducing another temporal upsampling stack or recomputing the
harmonic source.

The adapter's final projection is initialized to zero. The untrained adapter
therefore begins as an exact residual no-op, which avoids an abrupt generator
output change when evaluating the architecture from an existing checkpoint
copy.

## F0 and N Contracts

The decoder should return prepared F0 and prepared `N` alongside its synthesis
features and mask. Returning `N` makes the actual smoothed conditioning visible
to validation and diagnostics. It does not authorize direct generator-side
`N` injection.

`N` remains a deterministic frame-level log-energy curve. It is not Gaussian
excitation noise. It affects waveform generation through the repeatedly
conditioned decoder features, matching StyleTTS2.

F0 remains present in two differentiable roles:

1. Its projected curve is concatenated throughout the decoder, allowing
   decoder and supervised F0 gradients to reach the acoustic predictor.
2. Its prepared full-rate curve drives the harmonic generator under
   `torch.no_grad()`, matching StyleTTS2's source gradient contract.

The harmonic source retains:

- one fundamental and eight overtones;
- a 10 Hz voiced threshold;
- sine amplitude `0.1`;
- voiced Gaussian noise standard deviation `0.003`;
- unvoiced Gaussian noise standard deviation `0.1 / 3`;
- zero initial phase for the fundamental;
- random initial phases for overtones; and
- an explicit seeded `torch.Generator`.

## Training Conditioning

Architecture fidelity alone does not reproduce StyleTTS2's easier acoustic
training problem. Beetle currently reconstructs from F0 and `N` predicted from
a sampled posterior, so posterior and acoustic-prediction errors are amplified
by the decoder and harmonic source.

Use a reconstruction pretraining interval in which the decoder and generator
receive target F0 and target `N`. Train `FeatureLinear` against the same targets
in parallel, but do not use its predictions to condition waveform
reconstruction during this interval.

Before adversarial and feature-matching losses dominate, transition the
reconstruction path to predicted F0 and `N`. The transition schedule must be
explicit in configuration and validation must report which conditioning source
was used. Inference always uses predicted F0 and `N`.

This training policy adds no inference parameters or FLOPs. It first establishes
whether the decoder, harmonic source, multiband iSTFT, and PQMF path can produce
high-quality audio under correct conditioning, then trains the inference path
without permanently hiding FeatureLinear errors behind teacher conditioning.

## PQMF and Geometry

The synthesis geometry remains unchanged:

- sample rate: 24 kHz;
- decoder frame hop: 300 samples;
- temporal generator rate: 5;
- subbands: 4;
- subband iSTFT FFT size: 60;
- subband iSTFT hop: 15;
- frequency bins: `4 -> 8 -> 16 -> 31`; and
- final waveform length: `300T` samples.

PQMF filters, multiband iSTFT construction, magnitude/phase parameterization,
and output masking are outside the scope of this change.

## Complexity Requirements

The existing latent-to-audio preflight must remain strictly below the configured
15 GFLOPs per generated second ceiling. That ceiling is too broad to detect a
meaningful regression from this focused change, so validation must additionally
compare the result with the current approximately 3.94 GFLOPs/s reference.

The conditioning pyramid must satisfy both conditions:

- complete `FeatureLinear -> Decoder -> Generator` compute below 4.25 GFLOPs/s;
- no repeated harmonic waveform or STFT computation.

The added adapter should therefore cost less than approximately 0.31 GFLOPs per
generated second. Parameter count and measured FLOPs must come from the existing
complexity profiler rather than estimates.

## Validation

Temporary validation must run through the configured Beetle model path and
cover:

- decoder output geometry and masks with prepared F0 and prepared `N`;
- source adapter A and B shapes for padded and unpadded batches;
- identical harmonic features feeding both adapters;
- deterministic waveform output for equal source seeds;
- changed stochastic excitation for different source seeds;
- no harmonic-source gradient dependency on F0;
- gradients on both learned source adapters;
- exact waveform length of `300T`;
- unchanged four-band PQMF synthesis geometry;
- finite bfloat16-autocast forward and backward passes; and
- measured complexity below 4.25 GFLOPs per generated second.

Quality validation should compare the same held-out recordings under target
and predicted F0/`N`, using an identical posterior and harmonic-source seed.
Report 0-8 kHz log-mel error, speech-band magnitude correlation, and a metric
that measures harmonic-track continuity or instantaneous-frequency error.

## Non-Goals

- Do not add AdaLN or F0/`N`-generated normalization parameters.
- Do not feed `N` directly into the generator.
- Do not add style conditioning to Beetle's decoder or generator.
- Do not reproduce StyleTTS2's full-rate temporal upsampling stack.
- Do not change the harmonic count or excitation amplitudes.
- Do not change the multiband iSTFT or PQMF implementation.
- Do not relax the existing complexity gate.
