# Speaker Assignment Audit Report Design

## Scope

Build a domain-specific, package-local report generator that scans speaker assignment Parquet files without materializing the dataset. The core does not integrate with nodes, runtime persistence, database models, or schemas.

## Interface

The builder accepts assignment paths, an output directory, batch size, per-category limit, a synchronous cancellation callback, and a synchronous progress callback with `(processed_rows, total_rows)`. It returns a frozen typed result containing the JSON report, HTML report, and listening manifest paths.

The listening manifest has four typed categories:

- worst within-cluster rows, ordered by ascending best score;
- closest cross-cluster/impostor rows, ordered by descending second score;
- low-margin boundary rows, ordered by ascending margin;
- labeled rows assigned to clusters containing multiple true speakers.

Entries contain only assignment fields: segment ID, audio ID, duration, cluster identifiers, true label, and relevant scores. Start time is omitted because the assignment schema does not provide it.

## Data Flow

Parquet metadata provides the total row count. Cancellable batch generators feed `audit_metrics.compute_labeled_metrics` and each `audit_metrics.score_distribution` calculation in bounded sequential passes. A final scan maintains deterministic top-N heaps for the manifest. Heap ties use stable row data rather than encounter order, so shard or row reordering cannot change results.

Each batch invokes cancellation before processing and reports processed and total rows. Progress restarts for each bounded scan and ends at the metadata total.

## Artifacts

The output directory contains `audit-report.json`, `audit-report.html`, and `listening-manifest.json`. JSON preserves unavailable metric values as `null`. HTML is a small escaped summary of the same typed report. Files are replaced atomically from sibling temporary files.

## Validation and Testing

Batch size and per-category limit must be positive. Input Parquet files must expose the required assignment columns and compatible values; failures identify the missing columns or invalid setting.

Temporary strict-TDD tests cover typed paths and artifact content, explicitly unavailable metrics, category ranking and bounds, deterministic results after row reordering, and cancellation/progress per batch. Temporary tests are removed before the implementation commit.
