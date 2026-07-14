# Batched Smart Turn Node Design

## Goal

Add a batched runner node that classifies whether each incoming audio item is a complete conversational turn using the official Pipecat Smart Turn v3.2 CPU ONNX model. The node preserves every input item and exposes both the completion decision and its probability.

## Node contract

`SmartTurnPredict` belongs to the runner's `Audio` category and has these ports:

- `checkpoint`: broadcast `CheckpointRefPort` containing a `smart_turn` checkpoint.
- `audio`: `AudioPort`, batched normally by the runtime.
- `audio`: unchanged `AudioPort` output, one per input.
- `turn_complete`: boolean output, one per input.
- `probability`: float output in the inclusive range 0–1, one per input.

The completion decision uses the upstream threshold of `probability > 0.5`. The node does not filter items, mutate audio metadata, or write results to the database.

## Model acquisition

Extend `CatalogDownload` with a `turn_models` catalog. Its supported item is `pipecat-ai/smart-turn-v3`, resolved to the official `smart-turn-v3.2-cpu.onnx` file and stored as a checkpoint with type `smart_turn`. Workflows connect the catalog node's checkpoint output to `SmartTurnPredict`.

The catalog downloads only the required CPU model file. Inference never downloads weights implicitly, and PostgreSQL plus the existing asset CRUD remains the source of truth for checkpoint metadata and object storage.

## Runtime architecture

The node uses `BatchMode.MICRO_BATCH` and prepares the whole incoming batch before a single ONNX Runtime call. It keeps the ONNX session and Whisper feature extractor loaded across executions through `keep_loaded=True`. If the broadcast checkpoint changes, the node replaces the loaded session; teardown releases all model references.

Audio bytes already present on `Audio` are used directly. Missing bytes are retrieved in one bulk audio CRUD call. Each item is decoded to mono floating-point samples and resampled to 16 kHz. The last eight seconds are retained; shorter inputs are left-padded with zeroes so the most recent speech remains at the end, matching Smart Turn's upstream preprocessing. Whisper features are created for the full batch and passed to ONNX as one tensor.

The model's output length must equal the input batch length. Non-finite or out-of-range probabilities and empty audio inputs fail with an actionable error identifying the affected audio item. Cancellation is checked before audio preparation, between preparation chunks, and before inference. Progress reports prepared item counts for larger batches.

## Types and registration

Add a domain-agnostic `BoolPort` to the runner datatype registry rather than encoding the decision as an integer or JSON object. Register `SmartTurnPredictNode` through the runner registry and keep all Smart Turn-specific implementation under `src/runner/nodes/smart_turn/`. No Smart Turn or audio-specific behavior enters `src/runflow`.

## Dependency changes

Declare ONNX Runtime as a direct project dependency because the node imports it directly. Existing `transformers`, `librosa`, NumPy, Hugging Face Hub, shared asset helpers, and shared audio CRUD provide the remaining functionality. Update the lock through the repository's Nix-wrapped `uv` workflow.

## Validation

Use temporary tests during development and remove them before completion, per repository policy. Test-first coverage will establish:

- exact last-eight-second truncation and left padding;
- mono 16 kHz conversion and non-empty input validation;
- one batched inference call and one output per input;
- completion threshold behavior and probability validation;
- bulk loading of absent audio bytes;
- catalog checkpoint selection and runner/type registration.

Final validation runs repository checks through `nix develop --command ...`, starts or attaches to the shared development session, submits a small graph through `POST /graphs/runs`, and inspects it through the Nix-wrapped CLI. Any temporary workflow or test files are removed afterward.
