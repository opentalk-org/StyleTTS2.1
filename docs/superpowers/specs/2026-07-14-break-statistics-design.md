# Break Statistics Design

## Goal

Add dataset statistics for the number and duration of transcript break annotations without loading audio. The report must show how many `<break t=N>` tags each audio file contains and the distribution of all `N` values in milliseconds.

## Data source

Parse break annotations from each canonical segment's `text` field during dataset aggregation. Transcript text is authoritative because it represents the annotations consumed downstream; alignment entries may be absent or temporarily inconsistent with text.

Only exact tags shaped like `<break t=200>` count. Other angle-bracket text and malformed break tags do not contribute values.

## Aggregation

For every canonical segment:

- extract each break duration as a non-negative integer number of milliseconds;
- add the number of extracted tags to that segment's parent audio file;
- append every extracted duration to a dataset-wide duration collection.

Produce two numeric histograms with the existing adaptive histogram implementation:

- `break_count_per_file_histogram`: one value per audio file, including zero for files without breaks;
- `break_duration_ms_histogram`: one value for every valid break tag in the dataset.

The statistics payload version increases because the payload gains required fields. Both database-only and acoustic statistics modes use the same canonical segment records and therefore produce these histograms.

## Frontend

Extend `StatisticsPayload` with both histogram fields. Display a `Break annotations` histogram group in the audio-distribution area:

- `Breaks per file`, with unit `breaks` and count label `files`;
- `Break duration`, with unit `ms` and the default observation count.

The group remains visible for datasets with no breaks: the per-file histogram communicates that files contain zero annotations, while the duration histogram is empty.

## Validation

A temporary regression test will aggregate fixtures containing multiple segments, multiple files, valid breaks, no breaks, and malformed tags. It will verify per-file zero inclusion, exact tag parsing, duration pooling, and the payload version. Frontend type checking/build validation will verify the new payload fields and chart configuration. A registered statistics graph will validate the full runner path.

## Scope

This change does not modify break insertion, deduplication, stored segment schemas, histogram binning, or previously saved statistics payloads.
