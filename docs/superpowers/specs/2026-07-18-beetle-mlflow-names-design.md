# Beetle MLflow Names Design

## Scope

Simplify Beetle training metric names and validation artifact directory names.
The training loop, loss equations, validation contents, asynchronous reporting,
checkpoint state, and explicit sample ordering do not change.

## Metric contract

Metric names use one slash to create a useful top-level MLflow group. Losses
reported every optimizer step use `train/<name>`. Validation reports only the
mean across its configured recordings as `validation/<name>`.

Optimizer metrics use:

- `optimizer/<optimizer>_learning_rate`
- `optimizer/<optimizer>_amp_scale`
- `optimizer/<optimizer>_gradient_norm`

Module gradient norms use `gradient/<module>`. Existing `performance/<name>`,
`overhead/<name>`, and `system/<name>` metrics remain unchanged because each
already has a single meaningful group.

There are no epoch metrics and no `validation/sample/*` metrics. Per-sample
validation losses remain available in the validation manifest artifact rather
than becoming MLflow time series.

## Validation artifacts

Configured validation order determines one-based directory names:
`sample_1`, `sample_2`, and so on. UUIDs are excluded from directory names.
`metrics.json` retains each sample's one-based position, audio UUID, seed,
losses, and artifact names so every directory remains traceable to source data.

## Verification

Temporary tests will assert the exact training, optimizer, gradient, and
validation names; absence of per-sample MLflow metrics; one-based artifact
directories; and retained UUID lineage in `metrics.json`. The temporary tests
will be removed after verification in accordance with repository policy.
