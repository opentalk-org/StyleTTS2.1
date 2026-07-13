# Statistics Database Mode and Sampling Design

## Goal

Add database-only statistics and configurable random audio-reference sampling directly to a bounded, PostgreSQL-paged `AudioSource`. Both database and acoustic computation continue through the existing statistics aggregation and persistence pathway.

## Workflow Shapes

Acoustic mode uses:

`AudioSource → LoadAudio → LoadAudioSegments → AnalyzeAudioFeatures → AggregateDatasetStatistics → SaveStatisticsEntry`

Database-only mode uses:

`AudioSource → LoadAudioSegments → DatabaseStatisticsFeatures → AggregateDatasetStatistics → SaveStatisticsEntry`

The compute UI selects the graph shape. Both feature nodes emit the existing `feature_records` JSON port, so aggregation, saved entries, and the statistics viewer remain shared.

## Streaming Audio Source Sampling

`AudioSource` provides `ALL` and random-count modes while fetching lightweight references from PostgreSQL in bounded keyset pages.

- `ALL` emits each fetched page immediately under scheduler backpressure.
- Random-count mode scans bounded pages with reservoir sampling and retains at most the requested number of references.
- Source batch metadata reports the emitted count, preventing downstream aggregation from waiting for rows that sampling discarded.
- A requested count greater than the available count emits every available reference.
- Paging selects only reference fields and never loads audio bytes or segment JSON.

## Database Statistics Feature Node

`DatabaseStatisticsFeatures` accepts audio references after `LoadAudioSegments`. It builds feature records from fields already present in PostgreSQL-backed models:

- audio ID and name
- stored duration, sample rate, channel count, and metadata
- stored segments, transcript text, phonemes, alignments, speakers, and voices
- duplicate-segment collapse count produced by the shared segment-record conversion
- source batch metadata required by aggregation

It must not call `read_audio_file`, bulk audio-byte readers, pack storage, S3, waveform storage, librosa, or NumPy. Acoustic-only fields use empty values in the common record schema and the record explicitly marks acoustic metrics unavailable.

## Aggregation and Payload

`AggregateDatasetStatistics` continues to accept one feature-record format. It adds an `acoustic_metrics_available` flag to the saved payload. Counts, duration distributions, text/phoneme statistics, segment rates, speaker/voice summaries, and warnings work in both modes. RMS, clipping, frame-amplitude, and waveform-derived silence statistics are available only when acoustic feature records are present.

The payload also records the selected computation mode and sample scope so a saved entry can be interpreted later. Sampling applies to the entire result in both modes: `ALL` covers every source reference, while random `N` computes every displayed statistic from that selected subset.

## Compute UI

The statistics compute controls add:

- mode: `Database only` or `Analyze audio`
- sample scope: `ALL` or `Random sample`
- positive numeric sample count, shown only for random sampling

The workflow builder configures sampling on `AudioSource` and chooses the appropriate feature path. The entry name and dataset behavior remain unchanged.

When `acoustic_metrics_available` is false, the statistics viewer hides waveform-derived histograms and clipping values rather than rendering zero values as measured results. Database-derived duration and corpus sections remain visible.

## Registration and Errors

The database feature node and updated source are registered through the runner registry so their typed ports, settings, runtime defaults, and categories are available to the frontend schema.

The source validates a positive random count and fails clearly when its database page count changes unexpectedly. The database feature node fails clearly when an incoming value is not an unloaded audio reference with stored segments loaded through the normal pathway.

## Verification

Verification uses real graphs through `POST /graphs/runs` for database-only `ALL`, database-only random `N`, acoustic random `N`, and a requested count larger than the dataset. Database-only verification confirms no audio-byte read or storage access appears in node logs. Saved payloads must report the correct file count, sampling metadata, and acoustic availability.
