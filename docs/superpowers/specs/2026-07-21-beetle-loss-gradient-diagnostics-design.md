# Beetle Loss and Gradient Diagnostics

## Goal

Make Stage 1 loss balancing observable by attributing gradients to individual objectives and exposing spectral reconstruction detail that the current aggregate metrics hide.

## Cadence

Use one named Stage 1 diagnostics interval of 250 optimizer steps. Keeping this telemetry-only cadence outside the serialized training configuration preserves compatibility with checkpoints produced by the active process. Diagnostics run on every microstep belonging to a positive completed optimizer step divisible by 250, allowing the existing accumulator to average diagnostics correctly when gradient accumulation exceeds one.

The active training process is not restarted. It continues with its already-loaded code; the diagnostics apply to subsequent launches.

## Reconstruction Metrics

Preserve the existing three-resolution relative L1 reconstruction objective and its scalar value. Retain each resolution's loss and calculate relative L1 errors over mel bands whose center frequencies fall in 0–1 kHz, 1–4 kHz, 4–8 kHz, and 8–12 kHz. Average each band across the three resolutions.

Every diagnostic step reports the three resolution values and four frequency-band values. These are detached observations and do not change the optimized total.

## Loss-Attributed Gradients

Every diagnostic step, calculate weighted gradients without accumulating them into `.grad`:

- reconstruction, generator adversarial, and feature matching with respect to the synthesized waveform;
- the same three objectives with respect to generator parameters;
- F0 and N objectives with respect to FeatureLinear parameters;
- encoder KL with respect to audio-encoder parameters.

Report L2 norms. Report cosine similarity between reconstruction and adversarial waveform gradients and between reconstruction and feature-matching waveform gradients. Scheduled weights are applied before measuring, so a disabled objective reports zero influence.

Use `torch.autograd.grad(..., retain_graph=True)` and discard diagnostic gradients before the normal combined backward pass. Branched modules may contain parameters unused by one objective; exclude those `None` entries but require at least one gradient for each declared objective-to-target relationship.

## Clipping Metrics

Keep the existing pre-clipping optimizer and module norms. On diagnostic steps, additionally report the actual global clipping coefficient and a numeric clipped flag for each optimizer. The coefficient is `min(1, maximum_norm / (pre_clip_norm + epsilon))`.

## Reporting

Metric names remain generic training telemetry and flow through the existing callbacks and MLflow reporter. Diagnostic names are stable across diagnostic steps and are absent on ordinary steps.

## Validation

Temporary tests verify band assignment, per-resolution aggregation, weighted gradient norms and cosines, zero-weight behavior, due-step cadence, and clipping coefficients. Repository policy requires removing those temporary tests before completion. A real smoke graph is not necessary because this is standalone Beetle training rather than a runner node invocation; configuration parsing and a small tensor-level training diagnostic exercise provide focused verification.
