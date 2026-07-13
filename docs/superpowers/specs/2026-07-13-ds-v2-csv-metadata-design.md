# ds_v2 CSV Metadata Import Design

## Goal

Make `HetznerDsV2ParquetAudioSource` read audio bytes from the selected Hetzner
Parquet file and read all descriptive metadata from its corresponding CSV in
`/home/ds_v2_metadata`. Local imports and uncached downloads are removed.

## Source resolution

The node remains configured with the Hetzner host and remote Parquet path. It
derives one exact metadata path from the Parquet basename:

```text
/home/ds_v2/foo_processed.parquet
/home/ds_v2_metadata/foo_processed_metadata.csv
```

Both files are downloaded over SFTP into the runner's Hetzner cache. Cached
files are reused. A missing or empty Parquet or CSV fails the node with both
resolved remote paths in the error.

The `source`, `local_parquet_path`, and `cache_download` settings are removed.
The node always uses cached Hetzner SFTP input.

## Row ownership and validation

Parquet owns the encoded `audio` value. CSV owns transcript text, word
timestamps, MOS score, source fields, speaker identity, chunk/sample timing,
and source metadata.

The importer validates the pair before emitting audio:

- the CSV header contains every metadata field required by the importer;
- Parquet and CSV contain the same number of rows;
- corresponding rows match on stable identity fields: `chunk_index`,
  `sample_index`, `sample_start`, and `speaker_id`;
- the requested row offset and limit select corresponding rows from both files.

There is no fallback to Parquet metadata. Missing headers, malformed CSV,
different row counts, or the first identity mismatch raise an actionable error
that names the files, row index, field, and conflicting values.

## Runtime structure

The current ds_v2 importer exceeds the repository's file-size limit. The change
will separate remote file resolution, caching, CSV parsing, and pair validation
from conversion of validated rows into `Audio` and `AudioSegment` objects. The
registered node type, output port, batching behavior, voice creation, and
transcript-segment behavior remain unchanged.

## Verification

Validation will use a temporary graph with the existing ds_v2 workflow and the
known matching Hetzner pair. Failure checks will cover a missing derived CSV and
a row identity mismatch without committing permanent tests.
