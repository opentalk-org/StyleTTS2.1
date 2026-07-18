# Beetle Stage 1 Context Windows Design

## Goal

Train Stage 1 with fixed per-GPU tensor geometry while covering every usable
part of each source segment instead of encoding a full segment and selecting
one random crop. Decoder, Generator, and both discriminator families always
receive coherent 0.8-second targets.

## Window geometry

One Stage 1 item contains 32 posterior frames, 64 hop-300 acoustic frames, and
19,200 waveform samples. Source segments are traversed as consecutive
32-posterior-frame windows. When a segment has a remainder, one final window is
aligned to its end; this overlaps the preceding window rather than padding or
discarding the tail. At most one mel frame is excluded when the source length
is odd because posterior stride-two alignment requires an even mel boundary.

The posterior encoder has a 132-mel-frame receptive field. Each target window
therefore receives 66 mel frames of left context and 66 mel frames of right
context, giving a fixed 196-frame encoder input. Context outside the source
segment is represented by the same zero mel values and false mask used by
full-segment collation. Posterior frames 33 through 64 inclusive are the 32
target frames. Their mean and log scale must match the corresponding positions
from full-segment encoding in evaluation mode.

## Data planning and sharding

Stage 1 has a dedicated hierarchical planner. It continuously permutes eligible
source segments, assigns each whole segment to exactly one data-parallel rank,
expands it into ordered window descriptors, and maintains a small pending
window queue per rank. Each call removes exactly `stage1.batch_size` descriptors
from every rank queue, so all ranks receive disjoint windows and identical
batch counts. Planner checkpoints contain the source permutation, pending
descriptors, and batch position, making resume exact without materializing all
dataset windows in memory.

The Stage 1 source reads each distinct full source segment at most once per
prefetched batch. The collator preprocesses each distinct clip once, then
extracts every requested target waveform, target mel, contextual encoder mel,
and context mask. This keeps fixed accelerator shapes while reusing CPU decode
and mel work for adjacent windows from the same segment.

## Training path

Stage 1 no longer selects `AlignedSegments`. Both discriminator and generator
passes encode the fixed 196-frame contextual mel batch, slice the central
32-frame posterior, and run FeatureLinear, Decoder, and Generator over the
fixed target geometry. The discriminator receives the matching contiguous
19,200-sample real waveform.

Loss support is fixed and explicit:

- posterior KL uses the central 32 latent frames;
- F0 and N use the corresponding 64 target mel frames;
- reconstruction, adversarial, and feature-matching losses use the matching
  19,200 target samples;
- F0 extraction receives only the target mel supplied to Generator.

Stage 1 item throughput counts windows, so `batch_size` consistently means the
number of 0.8-second training windows per GPU. Tensor shapes and peak model
activation geometry do not depend on source duration.

## Isolation from later stages

Stage 2 keeps its existing full-utterance data pipeline. Its target, context,
style, and voice posterior calls remain frozen and enclosed in
`torch.no_grad()`.

Stage 3 remains unchanged: it keeps the existing full-utterance BeetleBatch,
random `adversarial.segment_samples` crop path, and current trainer behavior.
The contextual-window configuration applies only to Stage 1.

Stage 1 validation also remains full-utterance so saved reconstruction
artifacts continue to represent complete validation recordings.

## Verification

Temporary checks must demonstrate the old random-crop path first, then verify:

- every usable frame is covered by sequential/end-aligned window descriptors;
- rank queues are disjoint, fixed-size, deterministic, and exactly resumable;
- repeated windows from one segment trigger one source decode per batch;
- contextual and full-segment posterior mean/log-scale values match centrally;
- Stage 1 produces `[B,192,32]`, `[B,512,64]`, and `[B,1,19200]` geometry;
- generator and discriminator backward passes update their intended modules;
- Stage 2 full-audio `no_grad` behavior and Stage 3 source remain unchanged;
- project Python and smoke commands run through the Nix development shell.

Temporary verification files are removed before completion.
