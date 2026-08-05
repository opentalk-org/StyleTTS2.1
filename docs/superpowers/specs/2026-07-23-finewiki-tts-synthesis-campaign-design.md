# FineWiki TTS Synthesis Campaign

## Goal

Synthesize every line in `imports/tts_text_data/output` through registered
backend graph nodes, store every successful WAV as an audio record, and attach
it to a dataset named `tts_<engine>`. The campaign must preserve all unrelated
audio, resume without duplicating completed utterances, and target a total
runtime below five hours.

## Corpus and engine assignment

The input contains 101,250 lines in 741 logical voice files. Logical streams do
not imply distinct model checkpoints: the Piper catalog contains 101 models in
the selected languages and no Japanese model.

Assignment is deterministic and keeps every TXT file intact:

- Piper receives all `piper` streams except Japanese, plus `registered` streams
  in German, Dutch, Russian, Polish, Arabic, Turkish, and Korean.
- Kokoro receives `registered` streams in English, Spanish, French, Hindi,
  Italian, Japanese, Portuguese, and Mandarin, plus Japanese `piper` streams.

Piper selects one catalog model per language, preferring the model with the
largest speaker inventory. Logical streams rotate through model speaker IDs.
Kokoro rotates logical streams through presets whose prefix matches the
language. This assignment covers every corpus line exactly once.

The slower Chatterbox, F5-TTS, Orpheus, Dia, Fish Speech, and Raon engines are
not assigned corpus streams. On a single accelerator their sequential runtime
would violate the hard campaign deadline.

## Graph architecture

Two input/synthesis nodes own the long-running sources:

- `PiperCorpusSynthesis` reads its assigned TXT files, uses bounded CPU worker
  shards, and emits stored-audio payloads in batches.
- `KokoroCorpusSynthesis` loads one Kokoro lifecycle runtime, reads its assigned
  files, and emits bounded GPU synthesis batches.

Both branches feed the generalized `SaveAudioRecord` node. That node gains an
optional `dataset_id`; when set, audio creation and dataset membership commit in
the same database transaction. The campaign graph always sets it.

The graph contains no delete node. It cannot remove or detach existing audio.

## Durable identity and resume

Every output carries these annotation metadata fields:

- `tts_source_key`: stable `<engine>:<stream>:<sentence-index>` identity;
- `tts_dataset`: exact `tts_<engine>` dataset name;
- `engine`, `voice`, `language`, `text`, `stream`, and `sentence_index`.

Before planning work, each corpus node reads existing `tts_source_key` values
from its configured dataset through the shared dataset CRUD facade. Completed
keys are skipped. A failed or stopped graph can therefore be resubmitted
without duplicating successfully committed audio.

Audio names are `<stream>-<sentence-index>.wav`. Dataset membership is the
authoritative grouping, while `tts_dataset` provides a redundant per-audio
tracking flag.

## Throughput

Piper work is distributed across 15 balanced worker shards. Each ONNX runtime
uses one intra-op thread so parallel sessions do not oversubscribe the host.
Workers keep their current language model loaded and process contiguous
utterance groups.

Kokoro keeps its model loaded for the complete node lifecycle. The Piper and
Kokoro graph branches may run concurrently because they request disjoint CPU
and accelerator resources.

A 90-line real-graph benchmark is required before the complete launch. The
campaign proceeds only when measured throughput projects below five hours;
otherwise worker count or batch sizing is tuned without dropping lines.

## Validation

Temporary tests cover:

- deterministic engine and voice assignment;
- exact 101,250-line coverage with no duplicate source keys;
- resume filtering;
- atomic audio-record and dataset-membership persistence;
- dataset metadata flags;
- Piper shard balancing and one-thread ONNX sessions.

Node validation uses a real graph submitted through `POST /graphs/runs`.
Completion is proven from PostgreSQL by checking:

- `tts_piper` plus `tts_kokoro` contain 101,250 memberships;
- every member has the matching `tts_dataset` flag;
- source keys are unique and equal the corpus plan;
- audio byte lengths and durations are positive;
- no unrelated dataset membership or audio record was removed.
