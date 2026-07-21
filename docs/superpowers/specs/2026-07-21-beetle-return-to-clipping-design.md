# Beetle Stage 1 Return to Clipping

## Goal

Remove the active no-gradient-clipping ablation and start a clean clipped LJSpeech Stage 1 run from optimizer step zero.

## Design

Gracefully stop the process using `config-kl-off-no-clip.yaml`, then delete only its `output-kl-off-no-clip` directory and soft-delete its MLflow run. Preserve the older clipped `output-kl-off` baseline because it is not the active run and was not requested for deletion.

Launch the original `config-kl-off.yaml`, which sets both generator and discriminator `maximum_gradient_norm` to `10.0`, without `--resume`. Use a new empty `output-kl-off-clipped-fresh` directory so no checkpoint or optimizer state can be inherited.

## Verification

Confirm the removed process exits, the no-clipping output directory no longer exists, and its MLflow run is deleted. Confirm the replacement process is owned by `user`, uses the clipped configuration and new output directory, and creates a distinct MLflow run beginning at step zero. At step 250, confirm gradient clipping telemetry is present and any norm above 10 has a coefficient below one and `was_clipped` equal to one.
