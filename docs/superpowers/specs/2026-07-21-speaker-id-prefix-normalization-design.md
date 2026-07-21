# Speaker ID Prefix Normalization

## Scope

Normalize the only two imported datasets whose speaker IDs are not namespaced:

- CREMA-D: `1005` becomes `crema_d_1005`.
- EmoDB: `03` becomes `emodb_03`.

Already-qualified speaker IDs from every other dataset remain unchanged.

## Staged import generation

The CREMA-D and EmoDB preparation code will add the dataset slug when it creates
each staged `speaker_id`. Their current `data.json` manifests will be updated to
the same values so the next import is correct without rerunning downloads.

The shared uploader will continue to preserve staged speaker IDs. Prefixing does
not belong there because dataset adapters already emit qualified IDs using valid
dataset-specific conventions that do not always equal the directory slug.

## Existing backend rows

Rename the 91 CREMA-D speakers and 10 EmoDB speakers through the public shared
speaker CRUD. Each rename updates both the audio-file `speaker_id` column and
speaker IDs embedded in segment annotations. Dataset membership determines the
target prefix; no global numeric-ID rewrite is used.

## Verification

Before mutation, capture audio-file and segment totals for both datasets. After
mutation, verify:

- the totals are unchanged;
- CREMA-D has no numeric-only speaker IDs;
- EmoDB has no numeric-only speaker IDs;
- all 101 replacement IDs have the expected prefix;
- no speaker ID belongs to multiple datasets;
- a fresh staged payload produced by each adapter contains prefixed IDs.

No audio bytes, packs, waveform objects, empty segments, or unrelated datasets
are modified.
