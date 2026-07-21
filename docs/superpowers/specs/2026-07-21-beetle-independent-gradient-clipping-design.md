# Beetle Independent Gradient Clipping

## Goal

Prevent a high-gradient module from reducing updates for unrelated modules that share an optimizer. Apply the same clipping model to Beetle Stages 1, 2, and 3 without changing optimizer state, learning-rate schedules, loss weights, or checkpoint schemas.

## Design

Each `ScheduledOptimizer` owns one or more named gradient groups. A group contains modules that should share one gradient-norm limit. The optimizer still owns all parameters and performs one AdamW step; only clipping changes from one optimizer-wide norm to one norm per named group.

Before every optimizer step:

1. unscale gradients;
2. calculate the aggregate optimizer norm for continuity with existing telemetry;
3. calculate each named group norm;
4. clip each named group independently to the optimizer's configured maximum norm;
5. perform the unchanged optimizer and scaler step.

The optimizer constructor validates that every trainable optimizer parameter belongs to exactly one named group and that no group crosses optimizer ownership. Missing, duplicated, or foreign parameters fail explicitly before training.

## Stage Groups

Stage 1 generator groups are `audio_encoder`, `feature_linear`, `decoder`, and `generator`. Its discriminator uses `discriminators`.

Stage 2 groups cover every trainable module: phoneme encoders, context encoders, conditioning, style encoder, voice encoder, duration predictor, latent flow, aligner, style auxiliaries, and voice auxiliaries.

Stage 3 combines the complete Stage 1 and Stage 2 generator groups and retains the independent discriminator group.

This grouping keeps closely related small heads together while isolating the major model families whose gradient scales differ substantially.

## Metrics

Keep `optimizer/<name>_gradient_norm` as the aggregate pre-clipping norm. Keep `gradient/<group>` as each group's pre-clipping norm.

Every 250 optimizer steps in all three stages, report:

- `gradient/<group>_clip_coefficient`;
- `gradient/<group>_was_clipped`;
- `optimizer/<name>_clip_coefficient` as the minimum coefficient among that optimizer's groups;
- `optimizer/<name>_was_clipped` when any owned group clipped.

The optimizer-level coefficient remains an overview; the group metrics identify which module caused it.

## Compatibility

Optimizer parameter ownership and AdamW state remain unchanged, so checkpoint serialization does not change. No configuration field or fingerprint changes. The current Stage 1 experiment will restart from zero after verification, but the implementation does not require a checkpoint migration.

## Validation

Temporary tests prove independent clipping, exact ownership validation, aggregate and group metric values, Stage 2 coverage, and Stage 3 combined coverage. Focused compilation and configuration checks run through the Nix development shell. Temporary tests are removed before completion.
