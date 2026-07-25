from dataclasses import fields

import torch
from torch import Tensor
from torch.nn import functional as F

from ...data.records import BeetleBatch
from ...data.validation_records import ValidationRecording


_TUPLE_FIELDS = (
    "sample_keys",
    "speaker_ids",
    "recording_ids",
    "style_group_ids",
    "voice_group_ids",
)
_GROUP_TENSOR_FIELDS = (
    "style_views",
    "voice_views",
    "style_view_lengths",
    "voice_view_lengths",
    "style_distances",
)


def merge_validation_recordings(
    recordings: tuple[ValidationRecording, ...],
) -> BeetleBatch:
    batches = tuple(recording.batch for recording in recordings)
    values: dict[str, object] = {}
    for field in fields(BeetleBatch):
        items = tuple(getattr(batch, field.name) for batch in batches)
        if field.name in _TUPLE_FIELDS:
            values[field.name] = sum(items, ())
        elif field.name in _GROUP_TENSOR_FIELDS:
            values[field.name] = _pad_cat_groups(items)
        else:
            values[field.name] = _pad_cat(items)
    values["recording_ids"] = tuple(
        recording.audio_file_id for recording in recordings
    )
    return BeetleBatch(**values)


def _pad_cat(values: tuple[Tensor, ...]) -> Tensor:
    rank = values[0].ndim
    maximum = tuple(max(value.shape[axis] for value in values) for axis in range(1, rank))
    padded = []
    for value in values:
        widths = []
        for axis in reversed(range(1, rank)):
            widths.extend((0, maximum[axis - 1] - value.shape[axis]))
        padded.append(F.pad(value, tuple(widths)))
    return torch.cat(padded, dim=0)


def _pad_cat_groups(values: tuple[Tensor, ...]) -> Tensor:
    rank = values[0].ndim
    maximum = tuple(max(value.shape[axis] for value in values) for axis in range(1, rank))
    padded = []
    for value in values:
        widths = []
        for axis in reversed(range(1, rank)):
            widths.extend((0, maximum[axis - 1] - value.shape[axis]))
        padded.append(F.pad(value, tuple(widths)))
    return torch.cat(padded, dim=0)
