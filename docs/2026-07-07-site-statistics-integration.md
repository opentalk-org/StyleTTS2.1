# Site statistics — backend + workflow integration

## Goal

Turn the mock "Statistics" screen into a real feature backed by a runflow
workflow: compute dataset-level statistics on the runner, persist them as a
`statistics_entries` row, expose them through the backend API, and render them
in the frontend (with the ability to delete an entry). The computation must run
end-to-end through the real backend/runner and produce numbers that are
*reasonable* on the real data.

## What already exists

- **DB layer** `src/shared/db/statistics/` — `statistics_entries` table
  (`id, name, dataset_id, payload jsonb, metadata jsonb, created_at`), plus
  `create/get/list` CRUD and `StatisticsEntryCreate/Read` schemas.
- **Runner nodes** `src/runner/nodes/statistics/`
  - `AnalyzeAudioFeatures` — per-file audio-signal features (RMS, clipping,
    silence, frame stats) from decoded audio bytes.
  - `AggregateDatasetStatistics` — barrier node that buffers per-file records by
    `source_batch_id` until `source_batch_count` is reached, then emits one
    aggregate payload (histograms, n-grams, per-speaker rollups, …).
  - `SaveStatisticsEntry` — writes the aggregate payload to `statistics_entries`.
- **Frontend** `src/frontend/src/features/statistics/` — `StatisticsScreen` +
  chart primitives (`Histogram`, `HBars`, `RankList`, `StatTile`, `ChartCard`),
  fully driven by **mock** data in `logic.ts`. No `api.ts`/`query.ts`.

## Gaps (what this change adds)

1. **No producer for text/speaker statistics.** `AggregateDatasetStatistics`
   has an optional `segment_records` input, but no registered node produced it,
   so every text/phoneme/speaker field came out empty.
2. **The hard case: duplicate segments.** In the real `hetzner` dataset every
   audio file carries **4 segments over the identical time span** — one per ASR
   model (`canary`, `parakeet`, `src`, `whisper`) — all with the same speaker.
   A naïve aggregation counts each file's text/duration **4×**, so speaker
   durations exceed real audio, char histograms are inflated, and n-grams are
   quadrupled. This must be de-duplicated.
3. **No backend API** to list/get/delete statistics entries.
4. **Frontend is mock-only** and its chart props don't match the payload shape.

## Design

### 1. Single-stream, aligned segment records (runner)

Runflow delivers a `PortMode.LIST` input by draining whatever packets are
buffered when a task fires; the aggregate's `segment_records` port is
*optional*, so a two-stream design can fire with features present but segments
absent (or vice-versa) and mis-align them. To avoid that entire class of bugs we
use **one stream**: `AnalyzeAudioFeatures` embeds each file's de-duplicated
speech segments directly into its per-file record under a `segments` key, and
`AggregateDatasetStatistics` reads segments back out of the same records. Because
features and segments travel in the *same packet*, they can never drift out of
sync, and the barrier (`source_batch_id`/`source_batch_count`) already carried by
the feature record governs both.

Graph:

```
AudioSource → LoadAudio → LoadAudioSegments → AnalyzeAudioFeatures
            → AggregateDatasetStatistics → SaveStatisticsEntry
```

`LoadAudioSegments` attaches DB segments onto the `Audio` object; a new helper
`runner/nodes/statistics/segments.py::speech_segment_records(audio)` collapses
duplicates and returns clean segment dicts that `AnalyzeAudioFeatures` embeds.

### 2. Duplicate-segment de-duplication

Within a single audio file, segments are grouped by their (start, end) span
(rounded). Segments that share a span are treated as competing transcripts of
the same speech and collapsed to **one canonical segment**:

- Honour the data's own hint: pick the segment whose `metadata.text_column`
  equals `metadata.preferred_text_column` (here `text_src` → the `src` model).
- Otherwise fall back to a fixed model priority `src > canary > parakeet >
  whisper`, then to the longest non-empty text.
- Phonemes: take the canonical segment's `phon`; if empty, borrow any sibling's
  non-empty `phon`.

Distinct spans (real diarization splits) are preserved — only same-span copies
collapse. The number of collapsed copies is reported as a diagnostic so the UI
can show that de-duplication happened.

### 3. Richer, "reasonable" payload (aggregate v9)

In addition to the existing histograms/n-grams the aggregate now emits headline
scalars and guards so nothing is `None`/nonsensical:

- `total_duration_seconds`, `file_count`, `segment_count`, `speaker_count`
- `mean_duration_seconds`, `median_duration_seconds`
- `total_char_count`
- `duplicate_segments_collapsed` (diagnostic for the hard case)
- `text_length_warnings` — files whose transcript is empty / shorter than
  `text_min_chars` / longer than `text_max_chars` (feeds the warnings banner)
- `phonemes_available` — false when the source has no phonemes, so the IPA tab
  degrades gracefully instead of rendering empty charts as if broken.

Empty phoneme statistics are *correct* here (the source segments carry no IPA);
they are surfaced as "not computed", not as a failure.

### 4. Backend API (`src/backend/statistics/`)

- `GET  /statistics` → lightweight summaries `{id, name, dataset_id,
  file_count, created_at}` (never ships the multi-MB payload in the list).
- `GET  /statistics/{id}` → full `StatisticsEntryRead` (payload included).
- `DELETE /statistics/{id}` → remove an entry (frontend delete).
- `delete_statistics_entry` added to the shared CRUD facade.

### 5. Frontend wiring

`statistics/api.ts` + `query.ts` follow the repo pattern (thin `backendRequest`
functions + TanStack Query hooks). `logic.ts` gains pure adapters that map the
real payload to the existing chart props (`{edges,counts}` → histogram bins +
axis labels; `[[label,count]]` → `HBars`/`RankList`; scalars → `StatTile`). The
screen lists entries from the backend, loads the selected entry's payload,
renders it, and deletes via the API. Empty state (no entries) and the empty IPA
tab are handled explicitly.

## Testing (reproducible, no harness)

Run the workflow directly against the backend with an `InlineGraphRunRequest`
POSTed to `/graphs/runs` (dataset = `hetzner`), poll `/runs/{id}/snapshot` to
completion, then read the saved `statistics_entries` row and assert the values
are sane: `total_duration_seconds` ≈ real audio duration (not 4×),
`duplicate_segments_collapsed` ≈ 3× file count, non-empty char n-grams,
histograms with non-degenerate spread, speaker rollups summing to the total.
