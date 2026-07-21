# Beetle Short-Segment Padding Design

## Goal

Remove the lower duration cutoff from Beetle datasets while retaining the configured maximum duration. Stage 1 must accept recordings shorter than its fixed window by zero-padding them without treating padding as real audio.

## Configuration and indexing

- Set the default lower duration to zero and retain the field while the active run's checkpoints depend on its current config fingerprint. Eligibility does not apply a lower cutoff.
- Pass `data.maximum_seconds` into `DatabaseSegmentIndex` and exclude only segments longer than that value.
- Pass the same maximum into conditional cut planning; aligned mid-sentence cuts need only be positive and no longer than the configured maximum.
- Remove hardcoded `1–45` validation from `PlannedExample`. Positive ordered ranges remain enforced by `CutRange`, while the index and cut planner enforce the configured maximum.

## Stage 1 padding

- A short segment produces one Stage 1 plan starting at latent frame zero.
- The loader copies available target mel frames and waveform samples into zero-filled tensors of the configured 0.8-second Stage 1 geometry.
- `frame_mask` covers the complete fixed Stage 1 window. The padded tail is therefore trained as silence and remains available to the fixed 0.4-second adversarial crop.
- The contextual encoder window continues to use its existing zero padding and encoder mask.

## Conditional-stage padding

- Conditional batches have a minimum padded tensor length of the same 0.8-second window, while their ordinary frame masks continue to identify real audio.
- Stage 3 adversarial crop selection treats only the required padded crop region as available. Synthesis still receives the ordinary mask, so generated padding is zero and is compared with zero target padding.

## Verification

Use temporary tests, removed before handoff, to establish that:

1. custom maximum duration controls index eligibility;
2. a segment shorter than one Stage 1 window receives one plan;
3. collation pads mel and waveform tensors to the fixed window and keeps the fixed frame mask valid;
4. the default configuration uses a zero lower duration while the active run configuration remains fingerprint-compatible.

The running training process is not restarted or modified by this source change.
