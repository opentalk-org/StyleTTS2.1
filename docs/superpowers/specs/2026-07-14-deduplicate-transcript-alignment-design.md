# Deduplicate Transcript-Aligned Output Design

## Goal

Every non-null alignment produced by `DeduplicateOverlappingSegments` must contain exactly one ordered entry for every whitespace token in the selected consensus transcription. No word from a losing transcript may leak into the result.

## Transcript skeleton

Consensus segment selection remains unchanged. Split the winning segment's stripped text on whitespace; those exact tokens, including capitalization and attached punctuation, form the output alignment skeleton.

For each cluster member with alignment, map its ordered words to the skeleton using dynamic-programming sequence alignment. Words match when their lowercase alphanumeric-normalized forms are equal. The mapping is order-preserving and one-to-one, so repeated words are matched by occurrence rather than collapsed by a global word set. Extra alignment words remain unmatched and are discarded.

## Timing selection

Each skeleton position collects the mapped alignment candidates from all members. Choose the candidate with the highest numeric `score`; a missing score ranks below a numeric score. On equal scores, prefer the winning segment's candidate, then preserve cluster-member order for deterministic output.

Copy the selected candidate's timing and auxiliary fields, but replace `word` with the skeleton's exact transcript token. Clamp timings to the winning segment bounds. Candidate starts must remain ordered; a selected candidate that would reverse ordering is not usable for that position and that token is interpolated instead.

## Interpolation

Interpolate every contiguous run without a usable candidate:

- A run between matched entries evenly partitions the non-negative span from the previous entry's end to the next entry's start.
- A leading run partitions `segment.start` through the first matched entry's start.
- A trailing run partitions the last matched entry's end through `segment.end`.
- When no token has a match, partition the full segment duration evenly.
- If bounding matched entries overlap and leave no positive span, emit zero-duration entries at an order-preserving position between their midpoints.

Each generated entry contains `interpolated: true`. The output segment metadata records `alignment_interpolated_words` as the number of generated entries.

## Invariants and scope

Before returning, require the output alignment words to equal `winner.text.strip().split()` exactly and require every timing to be finite, inside the segment, and ordered by start. Empty winning text produces `alignment=None`. The shared `MergeAlignment` node retains its current union behavior; transcript-constrained projection is specific to overlap deduplication.

## Verification

Temporary regressions cover losing-track extras, winner omissions, repeated words, punctuation/case preservation, higher-score selection, deterministic ties, leading/internal/trailing interpolation, all-unmatched text, overlapping timing anchors, and empty text. A registered graph using `DeduplicateOverlappingSegments` verifies the invariant through the real runtime. Temporary tests and graph requests are removed before completion.
