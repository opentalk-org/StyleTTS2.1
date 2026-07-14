# DS v2 Transcript-Guided Alignment Design

## Problem

`HetznerDsV2Source` receives a segment transcript in `text_parakeet`, chunk-level
word timestamps in `text_timestamps`, and a `sample_start..sample_end` window.
Words ending exactly at a sample boundary are not assigned consistently by the
upstream dataset: one segment includes such a word while another excludes it.
Timestamp overlap therefore cannot reconstruct both alignments correctly.

## Selection

Treat `text_parakeet` as the authoritative ordered word sequence. Collapse its
whitespace, split it into exact tokens, and find every contiguous sequence of
timestamp entries whose stripped words equal those tokens. Matching remains
case- and punctuation-sensitive.

When multiple sequences match, select the sequence minimizing:

`abs(candidate_start - window_start) + abs(candidate_end - window_end)`

Timestamp order resolves equal scores deterministically by retaining the first
minimum candidate.

## Output

Rebase the selected timestamps by the sample window start and clip them to
`0..window.duration`, preserving the existing alignment representation. Words
outside the window may therefore become zero-duration boundary entries when the
authoritative transcript includes them.

If timestamps are absent, malformed as a container, or contain no exact
contiguous transcript match, emit `alignment=None`. Do not raise a transcript
mismatch error. Invalid or incomplete sample-window coordinates remain errors.

## Verification

Temporary regression coverage will exercise:

- row 138, where the preceding `a` must not be selected;
- row 42, where boundary word `Na` must be selected;
- repeated identical transcript sequences, choosing the closest window;
- no exact sequence, producing `alignment=None`.

The registered `HetznerDsV2Source` will then be run through a real graph for the
known failing rows. Temporary tests and graph requests will be removed afterward.
