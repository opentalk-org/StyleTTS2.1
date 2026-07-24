import torch
from torch import Tensor

from ...models.conditional import ConditionalModels
from ..aligned_window import safe_context_mask
from .features import WaveformMelExtractor


def encode_text_context(
    models: ConditionalModels,
    tokens: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    safe_mask, available = safe_context_mask(mask)
    encoded = models.context_phoneme_encoder(tokens, safe_mask)
    return encoded * available[:, 0], available


def encode_audio_context(
    models: ConditionalModels,
    latent: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    safe_mask, available = safe_context_mask(mask)
    encoded = models.context_audio_encoder(latent, safe_mask)
    return encoded * available[:, 0], available


def encode_view_latents(
    models: ConditionalModels,
    mel_extractor: WaveformMelExtractor,
    waveforms: Tensor,
    lengths: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    groups, views, channels, samples = waveforms.shape
    flattened = waveforms.reshape(groups * views, channels, samples)
    flat_lengths = lengths.reshape(groups * views)
    mel = mel_extractor(flattened, flat_lengths)
    with torch.no_grad():
        posterior = models.audio_encoder(mel.values, mel.mask, generator)
    return posterior.latent, posterior.mask
