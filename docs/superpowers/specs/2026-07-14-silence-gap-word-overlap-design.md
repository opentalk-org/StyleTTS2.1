# Silence Gap and Word Overlap Design

## Goal

Avoid consecutive break tags caused by a brief breath splitting one pause, while rejecting detected silence that substantially covers aligned speech.

## Detection

`InsertSilenceBreaks` gains `max_silence_gap`, an integer millisecond setting with default `80` and minimum `0`. After fixed-window RMS detection merges contiguous silent windows, it also bridges neighboring silence intervals whose intervening non-silent gap is no greater than `max_silence_gap`. The merged interval spans from the first interval's start through the second interval's end, so the bridged breath contributes to the resulting break duration.

Gap bridging happens once at audio level before intervals are mapped to segments. Existing behavior allowing adjacent segments to receive their own clipped break remains unchanged.

## Word-overlap rejection

`InsertSilenceBreaks` gains `word_overlap_drop_ratio`, a float setting with default `0.5` constrained to `0.0..1.0`. For each segment with alignment, first clip a detected silence interval to the segment bounds. Intersect that clipped silence with every ordinary aligned word interval, excluding existing `<break t=N>` entries. Merge those intersections into a union so overlapping word timings are counted only once.

Calculate:

```text
word_overlap_ratio = union_intersection_duration / clipped_silence_duration
```

Drop the silence candidate when `word_overlap_ratio > word_overlap_drop_ratio`. A ratio exactly equal to the setting remains eligible. Setting it to `0.0` drops every interval with any word overlap; setting it to `1.0` disables overlap-based dropping. Apply this rejection before boundary selection, `min_break_time`, and `drop_prob`.

## Example

The intervals `3.92–4.20s` and `4.26–4.42s` have a 60 ms intervening breath and merge into `3.92–4.42s` with the default 80 ms gap. Their union overlap with the neighboring aligned words is about 52% of the merged interval, so the candidate is dropped and no consecutive break tags are inserted.

## Verification

Use temporary helper regression tests for gap values below, equal to, and above 80 ms; disabled bridging at zero; overlapping word intervals; exactly 50%; and greater than 50%. Verify the known stored audio row through a registered graph and confirm the `Czernichowskiego, a` boundary receives no break while other valid pauses remain eligible. Remove temporary tests and graph requests before completion.
