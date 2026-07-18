# Beetle Cropped F0 Targets Design

## Purpose

Stage 1 and Stage 3 synthesize only an aligned 9,600-sample segment during
training, but the frozen JDC F0 extractor currently processes the complete
808-frame padded mel input. F0 target extraction and supervision should cover
only the audio frames supplied to the decoder and waveform generator.

## Training data flow

Each generator pass continues to select one `AlignedSegments` crop per example.
The same crop must select all of the following:

- target mel frames and their frame mask for frozen F0 extraction;
- predicted posterior F0 used by the Stage 1 F0 loss;
- posterior and conditional predicted F0 used by the Stage 3 F0 loss;
- decoder and waveform-generator inputs, as already implemented.

The frozen extractor therefore receives 32 mel frames for the baseline
9,600-sample crop instead of the full 808-frame padded tensor. Its returned
target has the same frame geometry as the synthesized segment.

The N loss and encoder KL loss remain full-utterance objectives. They do not
require the frozen F0 model and are outside this change.

## Stage behavior

Stage 1 computes cropped F0 targets once in the generator phase and compares
them with the matching crop of the audio encoder's predicted F0.

Stage 3 computes cropped F0 targets once for its shared generator segment and
uses them for both posterior and conditional F0 predictions. Posterior and
conditional synthesis must keep sharing the same segment.

Discriminator passes do not compute acoustic targets and remain unchanged.
Stage 2 remains unchanged.

## Validation

Validation remains full-utterance. Validation synthesizes the complete target,
so full-mel F0 extraction already matches the audio supplied to its generator.

## Verification

Temporary checks must first demonstrate that Stage 1 and Stage 3 pass full mel
to F0 target extraction. After implementation, the same checks must show that
the extractor receives the selected segment frames and that predicted and
target F0 tensors use the same mask and geometry. The real JDC extractor must
accept the 32-frame baseline crop.

The Stage 1 FLOP profile must be repeated after the change to report the actual
reduction in F0-extractor and whole-step compute.
