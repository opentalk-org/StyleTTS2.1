# DS v2 Alignment Window Design

## Problem

`HetznerDsV2Source` rebases chunk-level Parakeet word timestamps to each selected sample, but currently uses the padded audio duration as the alignment cutoff. When padding extends beyond `sample_end`, the loader retains words belonging to the following sample even though they are absent from `text_parakeet`.

## Design

Derive the timestamp window from the row's source coordinates:

- The window starts at `sample_start - chunk_start` within the chunk-level timestamps.
- The window ends at `sample_end - chunk_start`.
- Keep only words that overlap this source window, rebase their timestamps by the window start, and clip them to `0..(sample_end - sample_start)`.
- Continue using the decoded audio duration for `Audio.end` and `AudioSegment.end`, preserving intentional leading or trailing audio padding as silence.

Rows that contain Parakeet timestamps must have complete and ordered `chunk_start`, `sample_start`, and `sample_end` coordinates. Missing or invalid coordinates are rejected with a row-specific error because rebasing cannot be correct without them.

## Validation and Errors

After extraction, verify that the ordered alignment words reproduce `text_parakeet` after collapsing whitespace. Punctuation and letter case remain significant because both fields originate from the same Parakeet result. Raise an actionable row-specific error when they disagree instead of emitting inconsistent aligned data.

## Verification

Use the known failing cached row where the transcript ends with `smoleńskiej,` and the next chunk word is `toczonej` as the regression case. Confirm that `toczonej` is excluded, the final alignment ends at the 4.64-second sample boundary, and the padded segment still ends at 4.94 seconds. Then submit a small real graph containing `HetznerDsV2Source` and `InsertSilenceBreaks` and inspect its run result through the CLI.
