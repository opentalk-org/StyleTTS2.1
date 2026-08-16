import torch
from torch import Tensor, nn
from torchaudio.models import Conformer

from .prosody import NumberEmbedder


class PhonemeEncoder(nn.Module):
    """Condition phoneme embeddings on fixed-length voice tokens."""

    def __init__(
        self,
        text_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.conformer_pre = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=1,
            depthwise_conv_kernel_size=31,
            use_group_norm=True,
        )
        self.conformer_body = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=num_layers - 1,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )

    def forward(
        self,
        voice_tokens: Tensor,
        text: Tensor,
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_count = voice_tokens.size(1)
        lengths = input_lengths + token_count
        joint = torch.cat((voice_tokens.transpose(-1, -2), text), dim=-1)
        joint, lengths = self.conformer_pre(joint.transpose(-1, -2), lengths)
        joint, _ = self.conformer_body(joint, lengths)
        voice_hidden = joint[:, :token_count]
        text_hidden = joint[:, token_count:].transpose(-1, -2)
        return voice_hidden, text_hidden


class LegacyJointVoiceEncoder(nn.Module):
    """Jointly encode mel frames and phonemes with the paper-era Conformer."""

    def __init__(
        self,
        mel_dim: int = 80,
        text_dim: int = 512,
        voice_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=1,
            depthwise_conv_kernel_size=31,
            use_group_norm=True,
        )
        self.conformer_body = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=num_layers - 1,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Linear(text_dim, voice_dim)

    def forward(
        self,
        mel: Tensor,
        text: Tensor,
        input_lengths: Tensor,
        max_size: int,
    ) -> tuple[Tensor, Tensor]:
        input_lengths = input_lengths.to(mel.device)
        text_return = text.new_zeros(text.size(0), text.size(1), max_size)
        text = text[..., : input_lengths.max()]
        mel = self.mel_proj(mel)
        mel_length = mel.size(-1)
        lengths = input_lengths + mel_length
        hidden = torch.cat((mel, text), dim=-1).transpose(-1, -2)
        hidden, lengths = self.conformer_pre(hidden, lengths)
        hidden, _ = self.conformer_body(hidden, lengths)
        hidden = hidden.transpose(-1, -2)
        mel_hidden = hidden[:, :, :mel_length]
        text_hidden = hidden[:, :, mel_length:]
        voice = self.out(mel_hidden.mean(dim=-1))
        text_return[:, :, : text_hidden.size(-1)] = text_hidden
        return voice, text_return


class VoiceEncoder(nn.Module):
    """Encode a mel prompt into voice tokens that condition spoken text."""

    def __init__(
        self,
        mel_dim: int = 80,
        text_dim: int = 512,
        voice_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        token_count: int = 32,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(
            mel_dim,
            text_dim,
            kernel_size=3,
            padding=1,
        )
        self.prompt_conformer_pre = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=1,
            depthwise_conv_kernel_size=31,
            use_group_norm=True,
        )
        self.prompt_conformer_body = Conformer(
            input_dim=text_dim,
            num_heads=num_heads,
            ffn_dim=text_dim * 2,
            num_layers=num_layers - 1,
            depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.num_time = token_count
        self.positions = nn.Embedding(token_count, text_dim)
        self.embedder = NumberEmbedder(features=text_dim)
        self.out = nn.Linear(text_dim, voice_dim)
        self.phoneme_encoder = PhonemeEncoder(
            text_dim=text_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        mel: Tensor,
        text: Tensor,
        input_lengths: Tensor,
        max_size: int,
    ) -> tuple[Tensor, Tensor]:
        input_lengths = input_lengths.to(mel.device)
        text_return = text.new_zeros(text.size(0), text.size(1), max_size)
        text = text[..., : input_lengths.max()]
        mel = self.mel_proj(mel)
        mel_lengths = input_lengths.new_full(
            (mel.size(0),),
            mel.size(-1),
        )
        indices = torch.arange(self.num_time, device=mel.device)
        positions = self.positions(indices).unsqueeze(0).expand(
            mel.size(0),
            -1,
            -1,
        )
        time = torch.arange(mel.size(-1), device=mel.device)
        mel_hidden = mel.transpose(-1, -2) + self.embedder(time)
        prompt_hidden = torch.cat((positions, mel_hidden), dim=1)
        prompt_lengths = mel_lengths + self.num_time
        prompt_hidden, prompt_lengths = self.prompt_conformer_pre(
            prompt_hidden,
            prompt_lengths,
        )
        prompt_hidden, _ = self.prompt_conformer_body(
            prompt_hidden,
            prompt_lengths,
        )
        pooled = prompt_hidden[:, : self.num_time]
        voice_tokens, text_hidden = self.phoneme_encoder(
            pooled,
            text,
            input_lengths,
        )
        voice = self.out(voice_tokens.mean(dim=1))
        text_return[:, :, : text_hidden.size(-1)] = text_hidden
        return voice, text_return
