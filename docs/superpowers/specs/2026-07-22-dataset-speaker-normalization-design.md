# Dataset and Speaker Normalization Design

## Goal

Normalize every live dataset name and prefix every populated audio-file speaker
identifier with its normalized dataset name. The operation must include datasets
created while earlier audits were running and must not modify audio bytes,
metadata, transcripts, or missing speaker identifiers.

## Dataset names

The canonical dataset name is derived from the current database value by:

1. trimming surrounding whitespace;
2. converting letters to lowercase;
3. treating every run of whitespace or `/` characters as one `_`; and
4. removing leading or trailing `_` characters.

Other punctuation remains unchanged. For example,
`BVCC / VoiceMOS Challenge 2022` becomes
`bvcc_voicemos_challenge_2022`.

The operation reads the complete dataset table inside its transaction. It does
not use a previously recorded dataset count or name list, so newly imported
datasets are included.

## Speaker identifiers

For an audio file with a populated speaker identifier, the canonical form is:

`<normalized_dataset_name>_<current_speaker_id>`

If the current identifier already begins with the exact canonical prefix, it is
left unchanged. This makes the operation idempotent and avoids values such as
`aesdd_aesdd_01`. Older shorthand remains part of the original identifier; for
example, an AISHELL identifier may become `aishell-3_aishell3_SSB0005`.

Null, empty, and whitespace-only speaker identifiers remain unchanged. Segment
JSON, metadata JSON, voice prompts, style prompts, scores, and accuracy values
are outside this operation.

## Persistence boundary

Add focused bulk-normalization functions to the shared dataset CRUD feature.
The functions use SQLAlchemy set-based statements and execute the dataset-name
and speaker-identifier updates in one database transaction. Callers do not issue
ad hoc SQL or materialize millions of audio rows in application memory.

## Preconditions and failure behavior

Before either update, the transaction must reject:

- two datasets that would receive the same canonical name;
- an empty canonical dataset name;
- an audio file assigned to more than one dataset; or
- a populated audio-file speaker identifier assigned to no dataset.

Any failed precondition or update rolls back the complete transaction. Dataset
names and speaker identifiers therefore cannot be left partially normalized.

## Verification

Capture live counts immediately before the transaction, then verify afterward:

- the total dataset and audio-file counts are unchanged;
- every dataset name equals its canonical form and remains unique;
- every populated speaker identifier begins with its dataset's exact canonical
  prefix;
- null and blank speaker counts are unchanged;
- a second dry calculation finds zero dataset or speaker changes; and
- repository checks for the changed Python modules pass through the Nix
  development environment.

The final report includes the live dataset count, renamed dataset count, updated
speaker count, already-canonical speaker count, missing-speaker count, and total
sample count.
