# LoadAudio Byte Normalization

## Goal

Make `LoadAudio` apply its configured sample rate and channel count to the encoded audio bytes so the output payload and `Audio` metadata always agree.

## Design

`LoadAudio` will decode each input payload as floating-point audio, convert its channel layout, resample when necessary, and encode the result as PCM WAV. Mono conversion averages every source channel. Expanding mono duplicates it across the requested channels. For other channel-count reductions, the converter will preserve the first requested channels rather than inventing an ambiguous mix; expansions will repeat the last available channel until the requested count is reached.

The conversion logic will live in a focused helper beside the node. The node will continue bulk-loading missing database bytes, then normalize every payload, including audio whose bytes were already attached. Its returned duration, byte length, sample rate, channels, and annotation metadata will describe the normalized payload.

Decode or encode failures will propagate as actionable errors. Empty decoded payloads are invalid. No source database object will be modified; normalization affects only the workflow item emitted by `LoadAudio`.

## Verification

A temporary test will first demonstrate that a stereo WAV at a non-target sample rate currently remains unchanged despite updated metadata. After implementation, it will verify the emitted WAV has the configured frame rate and channel count, its duration remains stable within resampling tolerance, and its `Audio` fields match the payload. Project checks will run through `nix develop --command ...`. The temporary test will be removed before completion, as repository policy forbids retaining tests unless requested.
