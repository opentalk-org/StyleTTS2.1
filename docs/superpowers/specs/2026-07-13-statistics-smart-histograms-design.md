# Statistics Smart Histograms and Count Scatters

## Goal

Make every statistics histogram informative when its values contain a dense cluster and a long tail. Add duration-versus-total-word and duration-versus-total-character scatter plots without changing the persisted statistics schema.

## Readable Histogram Utility

`src/runner/nodes/statistics/aggregate_helpers.py` owns one histogram utility used by every histogram produced by `AggregateDatasetStatistics`.

The utility filters non-finite values and retains the configured bin count. Automatically ranged histograms with at least 100 values display the first through ninety-ninth percentile range so a small number of extreme values cannot flatten the dense region. Values below or above that range are counted in explicit underflow and overflow buckets; they are never silently discarded or folded into a misleading ordinary bin.

Integer-valued inputs use one integer-aligned bin per value when the displayed integer span does not exceed the configured bin count; wider spans use the configured number of numeric bins. Empty, smaller-than-100, and constant inputs use their complete deterministic range rather than percentile trimming. Callers with explicit semantic ranges keep those bounds and the configured fixed bin count.

The output extends the existing histogram object with `underflow` and `overflow` counts. The Plotly renderer places nonzero tail buckets directly beside the ordinary bins with `< boundary` and `≥ boundary` hover labels. Ordinary bars retain numeric widths and exact range labels.

## Count Scatter Plots

The existing sampled speaking-rate rows contain `[duration, rate, total]`. The frontend projects those rows into two additional charts:

- duration on x and total words on y;
- duration on x and total characters on y.

The four existing rate charts remain. Entries whose sampled rows lack the third value render the existing empty-state message for the new projections rather than producing invalid points.

## Scope and Errors

The change is limited to statistics aggregation helpers and statistics frontend chart configuration. It does not alter `runflow`, storage schemas, database models, or unrelated workflow behavior.

Invalid histogram configuration fails clearly. For automatically ranged histograms, ordinary counts plus underflow and overflow equal the number of finite input values. Explicit-range histograms retain their current in-range counting contract, including the upper boundary.

## Verification

Temporary focused tests cover percentile range selection, tail counts, bounded ranges, integer edges, empty inputs, constant inputs, and count conservation. Frontend checks cover tail-bar rendering and the two additional scatter projections. All commands run through `nix develop --command ...`, and temporary tests are removed before completion as required by the repository instructions.
