# Beetle Stage 1 No-Gradient-Clipping Ablation

## Goal

Run a clean LJSpeech Stage 1 ablation with gradient clipping effectively disabled, while preserving the completed clipped run through step 10,000 as the comparison baseline.

## Design

Leave `output-kl-off` and its configuration unchanged. Gracefully stop the active process only after confirming its exact-resume checkpoint records optimizer step 10,000.

Create a separate configuration from `config-kl-off.yaml` and change only `maximum_gradient_norm` for both the generator and discriminator optimizers from `10.0` to `1.0e30`. This retains the tested optimizer path and raw pre-clip gradient diagnostics while making the clipping coefficient exactly `1.0` for all practical gradients. Keep the seed, model, data, KL-off loss weights, learning rates, schedules, batch size, and 4,000-step validation interval identical.

Launch without `--resume` into the new `output-kl-off-no-clip` directory. The new MLflow run and optimizer state must begin at step zero.

## Verification

Confirm the baseline checkpoint is step 10,000 and remains present. Validate the new configuration through the project loader, confirm the new process is owned by `user`, and verify its command points to the no-clipping configuration and output directory. Confirm a distinct MLflow run begins from zero and reaches step 250. At step 250, require the named gradient metrics to remain present and all group clipping coefficients and flags to report `1.0` and `0.0`, respectively.

## Failure Handling

If graceful termination does not complete, inspect process state before escalating the signal. If launch or early training fails, preserve the clipped baseline and the failed no-clipping artifacts for diagnosis; never resume the new ablation from the clipped checkpoint.
