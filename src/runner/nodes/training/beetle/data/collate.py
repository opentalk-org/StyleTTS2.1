from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..config.data import AugmentationConfig
from .audio import AudioPreprocessor, ProcessedAudio
from .records import BeetleBatch
from .source import FetchedBatch, FetchedEmbeddingGroup, FetchedExample


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class _PreparedExample:
    source: FetchedExample
    target: ProcessedAudio
    pre_audio: Tensor
    post_audio: Tensor
    phoneme_ids: Tensor
    text_ids: Tensor
    pre_text_ids: Tensor
    post_text_ids: Tensor
    style_prompt_ids: Tensor
    voice_prompt_ids: Tensor


@dataclass(frozen=True)
class _PreparedGroups:
    waveforms: Tensor
    lengths: Tensor
    distances: Tensor
    group_ids: tuple[str, ...]


class BatchCollator:
    def __init__(
        self,
        preprocessor: AudioPreprocessor,
        phoneme_tokenizer: Tokenizer,
        text_tokenizer: Tokenizer,
        augmentation: AugmentationConfig,
        languages: tuple[str, ...],
        stage: int,
        minimum_frame_count: int | None,
    ) -> None:
        self.preprocessor = preprocessor
        self.phoneme_tokenizer = phoneme_tokenizer
        self.text_tokenizer = text_tokenizer
        self.augmentation = augmentation
        self.language_ids = {
            language: index for index, language in enumerate(languages)
        }
        self.stage = stage
        self.minimum_frame_count = minimum_frame_count

    def collate(self, fetched: FetchedBatch) -> BeetleBatch:
        prepared = tuple(self._prepare_example(item) for item in fetched.examples)
        target_waveforms, waveform_lengths = _pad_waveforms(
            tuple(item.target.waveform for item in prepared)
        )
        mels, frame_lengths = _pad_mels(
            tuple(item.target.mel for item in prepared),
            self.minimum_frame_count,
        )
        target_waveforms = _fit_waveform_frames(target_waveforms, mels.shape[-1])
        phonemes, phoneme_lengths = _pad_ids(tuple(item.phoneme_ids for item in prepared))
        texts, text_lengths = _pad_ids(tuple(item.text_ids for item in prepared))
        pre_audio, pre_audio_lengths = _pad_waveforms(tuple(item.pre_audio for item in prepared))
        post_audio, post_audio_lengths = _pad_waveforms(tuple(item.post_audio for item in prepared))
        pre_text, pre_text_lengths = _pad_ids(tuple(item.pre_text_ids for item in prepared))
        post_text, post_text_lengths = _pad_ids(tuple(item.post_text_ids for item in prepared))
        style_prompt, style_prompt_lengths = _pad_ids(
            tuple(item.style_prompt_ids for item in prepared)
        )
        voice_prompt, voice_prompt_lengths = _pad_ids(
            tuple(item.voice_prompt_ids for item in prepared)
        )
        voice_groups = self._prepare_groups(fetched.voice_groups)
        style_groups = self._prepare_groups(fetched.style_groups)
        batch_size = len(prepared)
        max_phonemes = phonemes.shape[-1]
        max_frames = mels.shape[-1]
        return BeetleBatch(
            waveform=target_waveforms,
            mel=mels,
            phoneme_ids=phonemes,
            text_input_ids=texts,
            language_ids=self._resolve_language_ids(
                tuple(item.source.language for item in prepared)
            ),
            alignments=torch.zeros(batch_size, max_phonemes, max_frames),
            durations=torch.zeros(batch_size, max_phonemes),
            pre_audio=pre_audio,
            post_audio=post_audio,
            pre_text_ids=pre_text,
            post_text_ids=post_text,
            style_prompt_ids=style_prompt,
            voice_prompt_ids=voice_prompt,
            style_views=style_groups.waveforms,
            voice_views=voice_groups.waveforms,
            waveform_lengths=waveform_lengths,
            frame_lengths=frame_lengths,
            phoneme_lengths=phoneme_lengths,
            text_lengths=text_lengths,
            pre_audio_lengths=pre_audio_lengths,
            post_audio_lengths=post_audio_lengths,
            pre_text_lengths=pre_text_lengths,
            post_text_lengths=post_text_lengths,
            style_prompt_lengths=style_prompt_lengths,
            voice_prompt_lengths=voice_prompt_lengths,
            style_view_lengths=style_groups.lengths,
            voice_view_lengths=voice_groups.lengths,
            frame_mask=_length_mask(frame_lengths, max_frames).unsqueeze(1),
            phoneme_mask=_length_mask(phoneme_lengths, max_phonemes),
            text_mask=_length_mask(text_lengths, texts.shape[-1]),
            pre_audio_available=pre_audio_lengths > 0,
            post_audio_available=post_audio_lengths > 0,
            pre_text_available=pre_text_lengths > 0,
            post_text_available=post_text_lengths > 0,
            style_prompt_available=style_prompt_lengths > 0,
            voice_prompt_available=voice_prompt_lengths > 0,
            style_distances=style_groups.distances,
            view_seeds=torch.tensor([item.source.plan.seed for item in prepared]),
            sample_keys=tuple(item.source.plan.key for item in prepared),
            voice_ids=tuple(item.source.voice_id for item in prepared),
            recording_ids=tuple(item.source.plan.key.audio_file_id for item in prepared),
            style_group_ids=style_groups.group_ids,
            voice_group_ids=voice_groups.group_ids,
        )

    def _resolve_language_ids(self, languages: tuple[str | None, ...]) -> Tensor:
        if self.stage == 1:
            return torch.zeros(len(languages), dtype=torch.long)
        return torch.tensor(
            [self.language_ids[language] for language in languages],
            dtype=torch.long,
        )

    def _prepare_example(self, item: FetchedExample) -> _PreparedExample:
        target = self.preprocessor.decode(item.target_clip, item.plan.key)
        pre_audio = self._optional_audio(item.pre_clip, item.plan.key)
        post_audio = self._optional_audio(item.post_clip, item.plan.key)
        return _PreparedExample(
            source=item,
            target=target,
            pre_audio=pre_audio,
            post_audio=post_audio,
            phoneme_ids=_ids(self.phoneme_tokenizer, item.phonemes),
            text_ids=_ids(self.text_tokenizer, item.text),
            pre_text_ids=_ids(self.phoneme_tokenizer, item.pre_text),
            post_text_ids=_ids(self.phoneme_tokenizer, item.post_text),
            style_prompt_ids=_ids(self.text_tokenizer, item.style_prompt or ""),
            voice_prompt_ids=_ids(self.text_tokenizer, item.voice_prompt or ""),
        )

    def _optional_audio(self, clip, key) -> Tensor:
        if clip is None:
            return torch.zeros(1, 0)
        return self.preprocessor.decode(clip, key).waveform

    def _prepare_groups(
        self,
        groups: tuple[FetchedEmbeddingGroup, ...],
    ) -> _PreparedGroups:
        if not groups:
            empty = torch.zeros(0, 0, 1, 0)
            return _PreparedGroups(empty, torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0), ())
        prepared = []
        distances = []
        for group in groups:
            group_audio = []
            group_distances = []
            for view in group.views:
                audio = self.preprocessor.decode(view.clip, view.plan.key).waveform
                group_audio.append(
                    self.preprocessor.augment(audio, view.plan.seed, self.augmentation)
                )
                group_distances.append(view.plan.distance_seconds)
            prepared.append(tuple(group_audio))
            distances.append(tuple(group_distances))
        waveforms, lengths = _pad_group_waveforms(tuple(prepared))
        return _PreparedGroups(
            waveforms,
            lengths,
            torch.tensor(distances, dtype=torch.float32),
            tuple(group.group_id for group in groups),
        )


def _ids(tokenizer: Tokenizer, text: str) -> Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def _pad_ids(values: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([value.numel() for value in values], dtype=torch.long)
    maximum = int(lengths.max()) if len(values) else 0
    output = torch.zeros(len(values), maximum, dtype=torch.long)
    for index, value in enumerate(values):
        output[index, : value.numel()] = value
    return output, lengths


def _pad_waveforms(values: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([value.shape[-1] for value in values], dtype=torch.long)
    maximum = int(lengths.max()) if len(values) else 0
    output = torch.zeros(len(values), 1, maximum)
    for index, value in enumerate(values):
        output[index, :, : value.shape[-1]] = value
    return output, lengths


def _pad_mels(
    values: tuple[Tensor, ...],
    minimum_frame_count: int | None,
) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([value.shape[-1] for value in values], dtype=torch.long)
    maximum = int(lengths.max())
    padded_frames = maximum + maximum % 2
    if minimum_frame_count is not None:
        padded_frames = max(padded_frames, minimum_frame_count)
    output = torch.zeros(len(values), values[0].shape[0], padded_frames)
    for index, value in enumerate(values):
        output[index, :, : value.shape[-1]] = value
    return output, lengths


def _pad_group_waveforms(values: tuple[tuple[Tensor, ...], ...]) -> tuple[Tensor, Tensor]:
    view_count = len(values[0])
    if any(len(group) != view_count for group in values):
        raise ValueError("embedding groups must have equal view counts")
    lengths = torch.tensor(
        [[view.shape[-1] for view in group] for group in values],
        dtype=torch.long,
    )
    output = torch.zeros(len(values), view_count, 1, int(lengths.max()))
    for group_index, group in enumerate(values):
        for view_index, view in enumerate(group):
            output[group_index, view_index, :, : view.shape[-1]] = view
    return output, lengths


def _length_mask(lengths: Tensor, maximum: int) -> Tensor:
    return torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)


def _fit_waveform_frames(waveform: Tensor, frame_count: int) -> Tensor:
    target = frame_count * 300
    if waveform.shape[-1] == target:
        return waveform
    output = torch.zeros(waveform.shape[0], 1, target)
    copied = min(waveform.shape[-1], target)
    output[..., :copied] = waveform[..., :copied]
    return output
