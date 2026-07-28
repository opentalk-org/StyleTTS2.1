# FineWiki TTS Text Corpus Design

## Goal

Replace `imports/tts_text_data` with a deterministic generator that reads the
FineWiki-derived Parquets in `/workspace/lang-pl-bert` without modifying that
repository. The generator produces one normalized source-text file per TTS
voice while preserving the requested phoneme-length distribution.

## Corpus contract

The corpus contains 101,250 text lines across 741 voice files:

- 96 registered voices receive 450 lines each;
- 645 Piper voices receive 90 lines each;
- output covers English, German, French, Dutch, Chinese, Japanese, Hindi,
  Spanish, Portuguese, Italian, Russian, Polish, Arabic, Turkish, and Korean.

The language totals and registered-voice counts remain:

| Language | Lines | Registered voices | Piper voices |
| --- | ---: | ---: | ---: |
| en | 35,280 | 33 | 227 |
| de | 22,950 | 22 | 145 |
| fr | 12,960 | 12 | 84 |
| nl | 6,030 | 6 | 37 |
| zh | 5,220 | 5 | 33 |
| ja | 3,150 | 3 | 20 |
| hi | 2,970 | 3 | 18 |
| es | 2,790 | 3 | 16 |
| pt | 2,610 | 2 | 19 |
| it | 1,890 | 2 | 11 |
| ru | 1,260 | 1 | 9 |
| pl | 1,260 | 1 | 9 |
| ar | 990 | 1 | 6 |
| tr | 990 | 1 | 6 |
| ko | 900 | 1 | 5 |

Voice identities use `registered-<language>-<index>` and
`piper-<language>-<index>`. Every voice has its own UTF-8 TXT file with one
single-line FineWiki text span per line.

## Inputs and repository boundary

The generator reads:

- `/workspace/lang-pl-bert/data/parquet/<language>.parquet`, using its `lang`,
  `text`, and `phonemes` columns;
- `/workspace/lang-pl-bert/data/language_phonemes/<language>.txt`, used only to
  identify material already sampled by Lang-PL-BERT.

No file under `/workspace/lang-pl-bert` is created, changed, or deleted. The
selection and balancing code is copied and adapted into this repository so the
result does not depend on importing another checkout as a Python package.

## Prior-sample exclusion

Both stored Parquet phonemes and old TXT lines are normalized to NFC after
removing `<m/>` markers and collapsing whitespace. A source row is excluded
when its phoneme string has a similarity score of at least 0.70 against an old
line from the same language.

Matching uses an exhaustive all-pairs `rapidfuzz.fuzz.ratio` comparison within
each language. The compiled scorer processes Parquet batches as matrices, so
every pair at or above the configured threshold is excluded without relying on
a shortlist that could miss short or low-overlap strings.

The manifest reports the number of old lines, matched old lines, excluded
source rows, and unmatched old lines for every language. Unmatched old lines
are reported rather than silently treated as exclusions.

## Length balancing and selection

Selection uses the 32 phoneme-length bins from the Lang-PL-BERT TXT exporter:
1–15, then 16-character intervals through 496–512. Phoneme lengths determine
the bin, but the written value is normalized source `text`.

Each voice receives the most even feasible allocation across all 32 bins:

- a 450-line registered voice receives 14 lines per bin plus two rotating
  remainder assignments;
- a 90-line Piper voice receives two lines per bin plus 26 rotating remainder
  assignments.

Remainders rotate by voice index so aggregate language distributions do not
favor the earliest bins. Selection is deterministic under a fixed seed.

Rows are unique within each voice and preferred unique across its language.
If a bin cannot be filled uniquely, a second pass may reuse a row. A source row
may occur at most twice across the complete corpus. Generation fails with the
language, voice, and deficient bins if the exact contract cannot be satisfied.
Candidate collection uses a bounded priority heap per bin, retaining no more
than the aggregate bin requirement while still counting all eligible rows.

## Outputs

`imports/tts_text_data` is replaced by:

- focused generator modules and a command-line entry point;
- generated voice files grouped by language, kind, and bounded block
  directories;
- `manifest.json` containing the seed, inputs, language totals, voice records,
  exclusion statistics, bin counts, and reuse statistics.

Block directories hold at most 15 voice files so generated output respects the
repository folder-size convention.

Generation is atomic. It builds into a sibling temporary directory, validates
the complete corpus, then publishes with an atomic directory exchange when an
older output exists. A failure retains the staging directory for diagnostics
and does not publish a partial corpus.

## Validation

Temporary test-first checks cover:

- authoritative language, voice, and line counts;
- per-voice 450/90 line contracts;
- phoneme-bin quota rotation;
- normalized-text output rather than phoneme output;
- 0.70 exclusion behavior;
- deterministic selection;
- at-most-two corpus-wide reuse;
- clear failure on source shortage.

The production generation run is followed by a full manifest and file audit.
Temporary tests and scratch data are removed before handoff, following this
repository's rule against retaining tests unless requested.
