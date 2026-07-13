# Statistics Smart Histograms and Count Scatters

## Goal

Make every statistics histogram informative when its values contain a dense cluster and a long tail. Add duration-versus-total-word and duration-versus-total-character scatter plots without changing the persisted statistics schema.

## Smart Histogram Utility

`src/runner/nodes/statistics/aggregate_helpers.py` owns one data-adaptive histogram utility used by every histogram produced by `AggregateDatasetStatistics`.

The utility filters non-finite values and selects a bin width with the Freedman–Diaconis rule. The interquartile range determines local resolution, so extreme values do not flatten a dense cluster. The final bin count is the larger of `histogram_bins` and the data-derived count, capped at 200. When the interquartile range is zero, the utility uses the configured count. This hard limit keeps payload and rendering costs bounded.

Integer-valued inputs use integer-aligned edges when the data-derived width is near or below one unit. Empty and constant inputs retain deterministic non-zero ranges. Callers with explicit ranges keep those bounds, and values outside them retain the current include-or-clip behavior.

The output remains `{edges: number[], counts: number[]}`. Existing saved entries and the Plotly histogram renderer therefore require no schema migration or compatibility layer.

## Count Scatter Plots

The existing sampled speaking-rate rows contain `[duration, rate, total]`. The frontend projects those rows into two additional charts:

- duration on x and total words on y;
- duration on x and total characters on y.

The four existing rate charts remain. Entries whose sampled rows lack the third value render the existing empty-state message for the new projections rather than producing invalid points.

## Scope and Errors

The change is limited to statistics aggregation helpers and statistics frontend chart configuration. It does not alter `runflow`, storage schemas, database models, or unrelated workflow behavior.

Invalid histogram configuration fails clearly. Histogram counts continue to include each finite in-range value exactly once, including the upper boundary.

## Verification

Temporary focused tests cover long-tail resolution, bounded ranges, integer edges, empty inputs, constant inputs, and count conservation. Frontend checks cover the two additional scatter projections. All commands run through `nix develop --command ...`, and temporary tests are removed before completion as required by the repository instructions.
