from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import BeetleConfig
from .records import BeetleBatch, SegmentKey
from .validation_records import (
    PreparedValidationAudio,
    ValidationRecording,
)


class ValidationTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def collate_validation_recording(
    config: BeetleConfig,
    item: PreparedValidationAudio,
    phoneme_tokenizer: ValidationTokenizer,
    text_tokenizer: ValidationTokenizer,
) -> ValidationRecording:
    stored = item.stored
    processed = item.processed
    original_frames = processed.mel.shape[-1]
    dynamic_frames = original_frames + original_frames % 2
    padded_frames = dynamic_frames
    mel = F.pad(processed.mel, (0, padded_frames - original_frames)).unsqueeze(0)
    padded_samples = padded_frames * config.audio.hop_length
    waveform = F.pad(
        processed.waveform,
        (0, padded_samples - processed.waveform.shape[-1]),
    ).unsqueeze(0)
    text = " ".join(segment.text.strip() for segment in stored.segments).strip()
    phonemes = " ".join(
        segment.phonemes.strip() for segment in stored.segments
    ).strip()
    phoneme_ids = _ids(phoneme_tokenizer, phonemes, True)
    text_ids = _ids(text_tokenizer, text, True)
    style_prompt = _ids(text_tokenizer, stored.style_prompt or "", False)
    voice_prompt = _ids(text_tokenizer, stored.voice_prompt or "", False)
    language_id = config.architecture.language.values.index(stored.language)
    segment_id = stored.segments[0].segment_id
    key = SegmentKey(stored.audio_file_id, 0, segment_id)
    frame_lengths = torch.tensor([original_frames], dtype=torch.long)
    waveform_lengths = torch.tensor(
        [processed.waveform.shape[-1]],
        dtype=torch.long,
    )
    phoneme_lengths = torch.tensor([phoneme_ids.numel()], dtype=torch.long)
    text_lengths = torch.tensor([text_ids.numel()], dtype=torch.long)
    frame_mask = _length_mask(frame_lengths, padded_frames).unsqueeze(1)
    phoneme_mask = _length_mask(phoneme_lengths, phoneme_ids.numel())
    text_mask = _length_mask(text_lengths, text_ids.numel())
    prompt_style = style_prompt.unsqueeze(0)
    prompt_voice = voice_prompt.unsqueeze(0)
    views = waveform.unsqueeze(1).repeat(1, 2, 1, 1)
    view_lengths = waveform_lengths.unsqueeze(1).repeat(1, 2)
    speaker_id = stored.segments[0].speaker_id
    batch = BeetleBatch(
        waveform=waveform,
        mel=mel,
        phoneme_ids=phoneme_ids.unsqueeze(0),
        text_input_ids=text_ids.unsqueeze(0),
        language_ids=torch.tensor([language_id], dtype=torch.long),
        alignments=torch.zeros(1, phoneme_ids.numel(), padded_frames),
        durations=torch.zeros(1, phoneme_ids.numel()),
        style_prompt_ids=prompt_style,
        voice_prompt_ids=prompt_voice,
        style_views=views,
        voice_views=views.clone(),
        waveform_lengths=waveform_lengths,
        frame_lengths=frame_lengths,
        phoneme_lengths=phoneme_lengths,
        text_lengths=text_lengths,
        style_prompt_lengths=torch.tensor([style_prompt.numel()]),
        voice_prompt_lengths=torch.tensor([voice_prompt.numel()]),
        style_view_lengths=view_lengths,
        voice_view_lengths=view_lengths.clone(),
        frame_mask=frame_mask,
        phoneme_mask=phoneme_mask,
        text_mask=text_mask,
        style_prompt_available=torch.tensor([style_prompt.numel() > 0]),
        voice_prompt_available=torch.tensor([voice_prompt.numel() > 0]),
        style_distances=torch.zeros(1, 2),
        view_seeds=torch.zeros(1, dtype=torch.long),
        sample_keys=(key,),
        speaker_ids=(speaker_id,),
        recording_ids=(stored.audio_file_id,),
        style_group_ids=(str(stored.audio_file_id),),
        voice_group_ids=(speaker_id,),
    )
    return ValidationRecording(
        stored.audio_file_id,
        batch,
        processed.waveform.detach().clone(),
    )


def _ids(
    tokenizer: ValidationTokenizer,
    text: str,
    required: bool,
) -> Tensor:
    values = tokenizer.encode(text) if text else []
    if required and not values:
        raise ValueError("validation tokenizer returned no tokens")
    return torch.tensor(values, dtype=torch.long)


def _length_mask(lengths: Tensor, maximum: int) -> Tensor:
    return torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
