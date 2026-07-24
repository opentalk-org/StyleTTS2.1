# TTS Corpus Throughput Design

## Goal

Generate the remaining TTS corpora as quickly as the available accelerators allow while preserving high-tier models, one transcript segment per audio record, dataset lineage, and resumability.

The operational target is completion within five hours. The launcher must report when available hardware cannot sustain the required rate rather than presenting an unsupported ETA.

## Bottleneck

The current Chatterbox graph uses CUDA but averages about 0.18 items per second on an RTX 5090. GPU utilization is about 30%, VRAM use is about 6.5 GiB, and one CPU core is saturated. `EngineRuntime.synthesize_batch` loops over requests sequentially. Chatterbox also rebuilds the same reference voice conditioning for every sentence even though each stream contains 450 consecutive sentences.

## Design

Each clone runtime owns voice preparation. Chatterbox caches prepared model conditionals by a stable clone identity and reuses them for later requests. The cache is bounded to the active voice because corpus jobs are ordered by voice, so it does not retain all reference tensors on the GPU.

Engine-native batching remains an explicit runtime override. Orpheus continues using vLLM batch generation. Engines without a correct native batch API remain sequential inside a node; the scheduler must not describe this as model batching.

Corpus plans can be partitioned into deterministic shards. A shard selects jobs by stable plan position before completed `source_key` filtering. Separate graphs can therefore run on separate runners without overlap, while existing audio remains the source of resume state. Each shard uses the same destination dataset and preserves the existing `tts_<engine>` flag and transcript segment.

The campaign launcher measures completed records over a bounded real graph and compares the observed rate with:

`remaining records / remaining deadline seconds`

It may launch one shard per online accelerator runner. It must not launch multiple heavyweight replicas on one accelerator unless a model-specific VRAM benchmark establishes that the replicas fit.

## Validation

A real graph first generates a small uncached Chatterbox sample followed by multiple sentences from the same voice. The node log and graph performance report must show one reference-conditioning operation per voice and improved sustained throughput. Stored records must retain the expected dataset name, `source_key`, and transcript segment.

The full campaign is replaced only after the benchmark succeeds. Stopping the prior graph is safe because completed records are discovered through their durable source keys.

## Constraints

- Do not reduce model tier or synthesis quality settings.
- Do not create duplicate audio to inflate completion counts.
- Do not delete existing audio.
- Do not add audio-specific behavior to `runflow`.
- Run all project commands through `nix develop --command`.
