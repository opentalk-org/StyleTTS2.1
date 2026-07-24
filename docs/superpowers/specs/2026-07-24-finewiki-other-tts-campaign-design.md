# FineWiki Other-TTS Campaign Design

## Goal

Extend the existing FineWiki synthesis campaign to Chatterbox, F5-TTS,
Orpheus, Dia, Fish Speech, and Raon OpenTTS. Preserve every existing audio
record, store each engine in `tts_<engine>`, and make every run resumable.

## Voice and text assignment

The 96 `registered-<language>-<index>` streams remain the cloning-voice
inventory. Each stream keeps its existing 450-line TXT file.

Engine coverage follows the capabilities of the installed runtime:

| Engine | Voice source | Languages | Jobs |
| --- | --- | --- | ---: |
| Chatterbox | 96 registered references | all 15 corpus languages | 43,200 |
| F5-TTS | registered references | English and Chinese | 17,100 |
| Orpheus | 8 built-in presets | English | 3,600 |
| Dia | registered references | English | 14,850 |
| Fish Speech | 96 registered references | all 15 corpus languages | 43,200 |
| Raon OpenTTS | registered references | English | 14,850 |

The extension therefore contains 136,800 synthesis jobs. It does not reroute
or delete the 101,250 existing Piper and Kokoro records.

## Clone-reference selection

Every registered stream already has stored synthesis results in `tts_piper` or
`tts_kokoro`. Reference selection reads those datasets through shared database
CRUD facades and binds one stored clip back to the same stream.

A candidate must:

- have metadata `stream` equal to the registered stream identity;
- have the same language as the stream;
- contain one nonempty full-duration transcript segment;
- have stored audio bytes and a positive duration;
- be between 4 and 12 seconds long.

Selection is deterministic: minimize distance from eight seconds, then
sentence index, then audio UUID. A missing eligible reference is a hard error
that identifies the stream. Reference audio and transcript are loaded once per
stream and reused by compatible engine jobs. Piper LibriTTS voices are already
excluded from the source campaign, so no LibriTTS reference can be selected.

## Planning and durable identity

The corpus planner gains engine-specific plans without changing the generic
Runflow scheduler. Clone jobs carry the registered stream, reference audio ID,
language, sentence index, text, and engine. Orpheus jobs carry their preset
voice instead of a clone reference.

The source key includes the engine, voice or registered stream, sentence
index, normalized-text digest, and reference audio ID when cloning. Corrected
TXT content or a changed reference therefore produces a distinct key instead
of being incorrectly skipped by resume.

Every output contains a full-duration transcript segment and metadata for
`tts_source_key`, `tts_dataset`, engine, voice, reference audio ID, language,
text, stream, and sentence index.

## Runtime and graph execution

One reusable corpus node handles lifecycle loading, bounded job iteration,
resume filtering, cancellation, progress, and audio construction for the six
engines. Engine runtimes retain their specialized batching behavior.

Only one large-model engine runs on the accelerator at a time. Each engine is
submitted as a separate real graph with its own checkpoint and
`SaveAudioRecord` branch. Checkpoints are downloaded through `CatalogDownload`
before their campaign graph. A one-reference, one-sentence smoke graph must
succeed before the full resumable run is submitted.

No graph contains deletion or detachment operations. Existing datasets and
audio remain untouched.

## Validation

Temporary tests cover engine/language job totals, deterministic reference
selection, text-sensitive source keys, transcript segments, dataset flags, and
resume filtering. Node validation uses real submitted graphs.

For each completed engine, a database audit requires:

- dataset membership equals the planned source-key set;
- every source key is unique;
- every record has positive audio bytes and duration;
- every record has one full-duration transcript;
- every record has the matching `tts_dataset` flag;
- no pre-existing audio or dataset membership was removed.
