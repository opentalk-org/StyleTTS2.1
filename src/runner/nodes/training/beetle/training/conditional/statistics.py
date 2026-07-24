from dataclasses import dataclass

from torch import Tensor

from ...losses.conditional import ConditionalBatchStatistics
from ...models.modules.conditioning import ConditionKeep
from ..aligned_window import AlignedWindow


@dataclass(frozen=True)
class ContextAvailability:
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor


def conditional_batch_statistics(
    window: AlignedWindow,
    availability: ContextAvailability,
    sampled_keep: ConditionKeep,
    effective_keep: ConditionKeep,
    seconds_per_frame: float,
) -> ConditionalBatchStatistics:
    ranges = window.ranges
    requested = ranges.target_requested_lengths.float()
    source = ranges.target_source_lengths.float()
    return ConditionalBatchStatistics(
        target_seconds=(requested * seconds_per_frame).mean(),
        target_padding_ratio=1 - source.sum() / requested.sum(),
        full_audio_ratio=ranges.full_selected.float().mean(),
        pre_text_available_ratio=availability.pre_text.float().mean(),
        post_text_available_ratio=availability.post_text.float().mean(),
        pre_audio_available_ratio=availability.pre_audio.float().mean(),
        post_audio_available_ratio=availability.post_audio.float().mean(),
        pre_text_random_drop_ratio=_drop_ratio(sampled_keep.pre_text),
        post_text_random_drop_ratio=_drop_ratio(sampled_keep.post_text),
        pre_audio_random_drop_ratio=_drop_ratio(sampled_keep.pre_audio),
        post_audio_random_drop_ratio=_drop_ratio(sampled_keep.post_audio),
        pre_text_effective_drop_ratio=_drop_ratio(effective_keep.pre_text),
        post_text_effective_drop_ratio=_drop_ratio(effective_keep.post_text),
        pre_audio_effective_drop_ratio=_drop_ratio(effective_keep.pre_audio),
        post_audio_effective_drop_ratio=_drop_ratio(effective_keep.post_audio),
    )


def _drop_ratio(keep: Tensor) -> Tensor:
    return (~keep).float().mean()
