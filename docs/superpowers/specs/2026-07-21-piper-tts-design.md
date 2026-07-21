# Piper TTS Design

## Goal

Add Piper as a downloadable, multilingual TTS engine that follows the existing
workflow contract and supports large dataset-synthesis runs across many voices.
Each Piper model is one voice.

## User workflow

The primary workflow is:

1. Generate or load text items.
2. Select a filtered pool of Piper voices.
3. Download missing selected voices.
4. Synthesize each text through the selected voice pool.
5. Save the emitted audio and synthesis results with voice and language lineage.

Target-duration workflows divide requested hours evenly across selected
languages and then evenly across voices in each language. Duration allocation is
an upstream workflow responsibility; Piper synthesis keeps the same item-oriented
contract as the other TTS nodes.

## Voice catalog and storage

The Piper catalog is read from the `voices.json` source used by
`piper.ttstool.com`. Catalog records are represented with typed models and expose
the stable voice ID, language, locale, display name, quality, sample rate, and
model/config download information.

A downloaded voice is stored through `shared.db.assets.crud` as a checkpoint
folder containing both its `.onnx` model and `.onnx.json` configuration.
PostgreSQL remains the source of truth for its metadata and object keys. The
checkpoint metadata records the Piper voice ID and catalog attributes so a
downloaded model can be resolved without fetching the remote catalog during
inference.

Catalog discovery must report the complete remote catalog and distinguish
downloaded voices. Downloads are explicit and reusable; synthesis never silently
downloads a missing model.

## Nodes and ports

### Piper voice selection

`PiperVoiceSelection` is an input node in the TTS category. It exposes catalog
selection in node settings and emits the generalized JSON voice port already used
by the other TTS nodes.

Selection supports:

- language, locale, and quality filters;
- explicit voice IDs;
- all matching downloaded voices; and
- a reproducible random subset using count and seed.

The emitted value is a single Piper voice or a `tts_voice_batch`, matching the
existing fan-out convention. Each Piper voice entry carries its checkpoint
reference because its model and voice are the same asset. Empty selections and
selected-but-undownloaded voices fail with actionable messages.

Catalog listing and download controls use the existing checkpoint/catalog API
and query seams. The UI must virtualize the model list because the catalog is not
treated as a small fixed enum.

### Piper synthesis

`PiperSynthesis` has the same principal shape as the other synthesis nodes:

- inputs: `text`, `voice`;
- outputs: `audio`, `synthesis_result`;
- settings: output name and synthesis controls supported by Piper; and
- TTS category, batching, cancellation, progress, and resource metadata.

It intentionally has no separate checkpoint input. The selected Piper voice
contains the checkpoint reference for its model. Language comes from the model's
catalog/config metadata rather than a manually chosen synthesis setting, avoiding
invalid model/language combinations.

Outputs preserve one result per text/voice/sample combination and include engine,
voice ID, language, locale, quality, checkpoint ID, input text, sample index, and
sample rate in metadata.

## Runtime behavior

Within each incoming batch, synthesis expands voice batches using the existing
fan-out semantics, then groups requests by checkpoint. It loads one Piper model,
synthesizes every request for that voice, releases the model, and proceeds to the
next voice. This prevents repeated model loads while keeping memory bounded for
large multilingual voice pools.

The runtime checks cancellation between model groups and utterances. Progress is
reported as completed utterances out of total utterances. Model/config validation
happens before synthesis and identifies the failing voice and missing file or
metadata field.

Piper is expected to run on CPU by default and declares honest generic resource
requirements. Lifecycle and model switching remain behind the node/runtime layer,
not in the graph scheduler.

## Registration and discovery

Piper is registered in the TTS engine enum, engine loader, runner node registry,
and schema export path. Catalog metadata is exposed through the backend so the
frontend can list, filter, and download voices without hardcoded voice IDs.

The implementation will add a smoke workflow following the existing
`tts_openrouter_kokoro.json` structure:

`generated texts -> Piper voice selection -> Piper synthesis -> saved audio`

## Error handling

The feature fails clearly for an unavailable catalog, malformed catalog entries,
empty filtered selections, undownloaded selected voices, absent model/config
files, unsupported model configuration, and synthesis failures. It does not use
hidden fallback voices, languages, or downloads.

If catalog refresh is unavailable, already downloaded voices remain discoverable
from stored checkpoint metadata, but the UI reports that the remote listing could
not be refreshed rather than presenting cached data as current.

## Verification

Implementation follows a temporary test-first cycle covering catalog parsing,
filtering and deterministic selection, voice payload validation, checkpoint
grouping, and output lineage. Temporary tests are removed before completion in
accordance with repository policy.

Final verification uses the Nix development shell and a real graph submitted via
`POST /graphs/runs`. The run downloads a small Piper voice, synthesizes multiple
texts through the registered node, and verifies audio/result metadata through the
CLI run and log commands. The frontend build and relevant Python checks must also
pass.
