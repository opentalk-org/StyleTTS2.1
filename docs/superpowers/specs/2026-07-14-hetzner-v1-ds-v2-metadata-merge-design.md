# Hetzner v1 and ds_v2 Metadata Merge Design

## Goal

Enrich each long recording emitted by `HetznerDsV1ParquetAudioSource` with the
matching ds_v2 sample transcriptions, speakers, scores, and Parakeet word
alignment. Preserve ds_v1 as the source of the audio bytes and recording-level
provenance while projecting ds_v2 sample coordinates onto the long recording.

## Metadata discovery and matching

The node derives one ds_v2 metadata CSV from its configured ds_v1 Parquet path.
For a ds_v1 basename `<stem>.parquet`, the metadata path is
`/home/ds_v2_metadata/<stem>_processed_metadata.csv`. The CSV uses the existing
Hetzner cache and SFTP retry helpers and is read once per node lifecycle.

Rows are indexed by their normalized `audio_path` and `filename`. A ds_v1 row
matches with its `opus_file`, or with `<video_id>.opus` when `opus_file` is
absent. When both identifiers are present, they must describe the same recording.
The embedded ds_v2 `metadata.video_id` must also agree with the matched ds_v1
`video_id`. Identity conflicts fail with the ds_v1 row, ds_v2 row, and conflicting
keys in the error. A ds_v1 recording with no ds_v2 samples remains valid and is
emitted without segments.

## Key-aware metadata merge

The ds_v1 row remains authoritative for audio and import provenance. Its
`source`, `source_host`, `source_parquet_path`, `source_row_index`, source byte
length, decoded sample rate, channels, duration, and all other ds_v1 columns are
never overwritten by sample metadata.

The decoded ds_v2 `metadata` object contains recording-level fields such as
`video_id`, `title`, channel identifiers, upload date, and uploader identifier.
For each key, the node fills a missing or null ds_v1 value from ds_v2. Equal
overlapping values are accepted. Unequal identity values fail; unequal
descriptive values remain ds_v1-authoritative and are retained in the segment's
ds_v2 source metadata rather than overwriting the recording.

Sample-specific fields are not flattened onto the audio because many ds_v2 rows
map to one recording. The audio metadata gains only:

- `ds_v2_metadata_path`, identifying the contributing CSV;
- `ds_v2_sample_count`, counting matched sample rows.

Each generated segment records its ds_v2 row index, chunk and sample indices,
chunk, speaker, and sample boundaries, speaker identifier, MOS score, transcript
column, preferred transcript column, and decoded source metadata. This preserves
per-sample values without collision or duplication of the full ds_v1 metadata.

## Transcript segments and alignment

Every non-empty transcript variant in `text_src`, `text_whisper`,
`text_parakeet`, and `text_canary` becomes a separate `AudioSegment`, following
the existing ds_v2 transcript ordering and metadata conventions. Segment IDs and
lineage IDs are stable over the ds_v1 Parquet path, ds_v1 row, ds_v2 metadata
row, and transcript source. All segments refer to the ds_v1 `audio_file_id` and
use the decoded ds_v1 sample rate and channel count.

The segment begins at absolute `sample_start`. Its end is
`sample_start + duration`, capped at the ds_v1 recording duration. This retains
the trailing padding represented by the ds_v2 sample audio. Invalid, non-finite,
reversed, or out-of-recording sample windows fail explicitly. Segments are
ordered by sample start, sample index, then transcript source.

Only the Parakeet transcript receives word alignment. The existing
transcript-guided ds_v2 alignment logic selects the exact contiguous word
sequence and creates timestamps local to the unpadded sample window. The v1
merge then adds `sample_start` to every word start and end. The resulting word
times are absolute coordinates on the long recording and must remain ordered
inside `sample_start..sample_end`. A row without an exact Parakeet timestamp
sequence produces `alignment=None`, matching current ds_v2 behavior.

## Runtime behavior

Metadata collection remains generic I/O work under the node's existing resource
policy. The node checks cancellation between metadata loading, row matching, and
audio decoding. It reads the selected ds_v1 rows and the metadata CSV without
loading ds_v2 audio bytes. Voice creation is outside this change: segments carry
the ds_v2 speaker identifier, while `voice_id` remains unset.

The ds_v1 sample workflow adds `SaveAudioSegments` after `SaveAudioRecord` so the
new transcripts and alignments are persisted by the smoke graph.

## Validation

Temporary tests exercise metadata-path derivation, identifier matching,
recording-level fill-only merging, conflict rejection, unmatched recordings,
all transcript variants, stable ordering, absolute alignment projection, missing
exact alignment, and invalid timing windows. Temporary tests are removed before
completion in accordance with repository policy.

End-to-end verification runs the updated ds_v1 workflow through
`POST /graphs/runs`, then inspects the run and persisted segments through the CLI.
The selected long audio must retain ds_v1 metadata and bytes, expose the expected
ds_v2 sample count, and store Parakeet word timings at their absolute positions
within the long recording.
