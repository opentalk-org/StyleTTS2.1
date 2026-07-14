# Silence Break Insertion Design

## Goal

Add a runner node that detects silent intervals in audio and annotates aligned
segment text and alignments with `<break t=N>` tokens, where `N` is the detected
silence length in milliseconds. Segments without alignments remain unchanged.
Also change `PadSilence` to express its threshold in dBFS.

## Node contract

`InsertSilenceBreaks` lives in `runner/nodes/audio_segments/`, is registered in
the runner registry, and uses the existing `AudioPort` for input and output. It
processes a micro-batch, requires audio bytes, preserves the waveform, and
returns the audio with updated segments. Persisting those segments remains the
responsibility of `SaveAudioSegments`.

Settings:

- `silence_threshold: float`: linear RMS in the inclusive range `0.0..1.0`.
- `window_size: int`: fixed, non-overlapping RMS window size in milliseconds;
  must be positive.
- `min_break_time: int`: minimum inserted break duration in milliseconds after
  clipping to a segment; must be positive.
- `insert_at_start: bool`: allow a break before the first aligned word.
- `insert_at_end: bool`: allow a break after the last aligned word.
- `drop_prob: float`: independent probability in the inclusive range
  `0.0..1.0` of dropping each eligible break during each execution.

The node uses ordinary per-execution randomness. `drop_prob=0.0` keeps every
eligible break and `drop_prob=1.0` keeps none. Random decisions are negligible
relative to audio decoding and RMS calculation.

## Silence detection

Decode each audio payload once as floating-point samples and mix channels to
mono for analysis. Audio sample zero corresponds to `audio.start`.

Partition samples into consecutive, non-overlapping windows. The final partial
window is included. A window is silent when its RMS is less than or equal to
`silence_threshold`. Merge adjacent silent windows into maximal intervals and
clamp the final interval to the audio duration.

## Mapping silence to segments

Segments with `alignment is None` or an empty alignment pass through unchanged.
For each aligned segment:

1. Clip every detected interval to `[segment.start, segment.end]`.
2. Discard the clipped interval when its rounded duration is below
   `min_break_time`.
3. An internal candidate exists where the interval overlaps the temporal gap
   between two consecutive aligned words.
4. Select at most one eligible boundary across the enabled start, internal, and
   end candidates. Choose the boundary with the greatest overlap; stable
   chronological order breaks ties.
5. A start candidate exists where the interval overlaps the range from the
   segment start to the first aligned word, but only when `insert_at_start` is
   enabled.
6. An end candidate exists where the interval overlaps the range from the last
   aligned word to the segment end, but only when `insert_at_end` is enabled.

Start and end mapping is segment-local. Consequently, silence between adjacent
segments may produce an end break in the previous segment and a start break in
the next when both settings are enabled. Each copy is clipped independently to
its owning segment.

Apply `drop_prob` independently after candidates are selected. A kept interval
becomes a chronological alignment entry:

```json
{"word": "<break t=200>", "start": 1.4, "end": 1.6}
```

The duration is the clipped interval duration rounded to the nearest
millisecond. Insert the identical token into segment text at the corresponding
aligned-word boundary with single-space separation. Preserve all other text.
Aligned words are matched sequentially in the transcript so repeated words are
unambiguous; a mismatch fails with the segment identifier instead of silently
rewriting the transcript. An identical existing break at the same timing and
position is reused so executing the node again does not duplicate it.

## Validation and failures

Audio without bytes fails with the audio identifier. Alignment entries used as
words must contain `word`, `start`, and `end`, have ordered finite timings, and
remain within their segment. Invalid alignment or transcript matching fails
with an actionable error naming the segment. Cancellation is checked between
audios and segments.

## PadSilence schema change

Rename `PadSilenceSettings.silence_threshold` to `silence_threshold_db`, bounded
from `-80.0` to `0.0`. Convert it to a linear RMS threshold with
`10 ** (dBFS / 20)` for active-window comparison. The emitted metadata uses the
renamed setting. No compatibility alias or fallback is provided.

## Verification

Use temporary tests to cover RMS interval merging, final partial windows,
minimum duration after clipping, internal greatest-overlap selection, optional
start/end duplication across segments, random dropping at both extremes,
transcript insertion, alignment insertion, idempotence, and untouched unaligned
segments. Cover `PadSilence` dBFS conversion and trimming behavior.

Then run the registered node through a small real graph using the repository's
Nix-managed stack and inspect the run through the CLI. Remove temporary tests
and scripts before finishing.
