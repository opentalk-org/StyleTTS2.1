# Beetle Stage 1 FLOP Benchmark Design

## Scope

Benchmark only the current Stage 1 training path. Stage 2 and Stage 3 are not
loaded or measured.

## Workload

The benchmark uses the local LJSpeech Stage 1 configuration and real database
batches: batch size 64, 808 padded mel frames, BF16 autocast, and aligned
32-frame/9,600-sample adversarial crops. It restores the latest saved Stage 1
model state so the executed workload matches the stopped training run.

## Measurement

A temporary local harness builds the same Stage 1 models, optimizer, trainer,
and data pipeline as the production runner without creating an MLflow run or a
checkpoint. It reports:

- counted FLOPs for discriminator backward, generator backward, optimizer, and
  the complete optimizer step;
- warmed wall time and steps per second over multiple uninstrumented steps;
- achieved TFLOPS as `TFLOP/step * steps/second`;
- utilization against the RTX 5090 dense BF16 Tensor peak of 209.5 TFLOPS;
- configured batch geometry and peak allocated GPU memory.

`torch.utils.flop_counter.FlopCounterMode` provides operation counts. Timing
uses production Stage 1 compilation, excludes initial compilation and warm-up,
and synchronizes CUDA at interval boundaries.

## Lifecycle

The harness and any generated local benchmark files are temporary and removed
after the report. Training remains stopped, and the saved checkpoint and run
state remain unchanged.
