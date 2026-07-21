# Beetle Stage 1 KL-Off Ablation

## Goal

Run a clean LJSpeech Stage 1 ablation with the posterior encoder KL loss disabled, while preserving the current KL-on run as a comparison baseline.

## Design

Stop the active KL-on process through its existing graceful termination path so it can finish the current exact-resume boundary. Keep its configuration, checkpoints, validation artifacts, and MLflow run unchanged.

Create a separate run directory by copying the active Stage 1 configuration. Change only `losses.generator.encoder_kl.value` from `1.0` to `0.0`; retain the dataset, model architecture, seed, batch size, optimizer, evaluation interval, and all other schedules. Launch without `--resume` into an empty output directory so optimizer step, weights, and MLflow tracking begin from zero.

## Verification

Confirm that the old process exits, the new process is owned by `user`, and its command points to the KL-off configuration and new output directory. Confirm through MLflow or startup logs that the run starts at step 0 and reports an effective encoder KL weight of zero. Finally, observe at least one completed optimizer step without an error.

## Failure Handling

If graceful termination fails, inspect the process and logs before escalating the signal. If the fresh launch fails, leave the preserved baseline untouched, diagnose the launch error, and do not resume from its checkpoints.
