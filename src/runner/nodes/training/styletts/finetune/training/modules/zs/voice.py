import torch
from torch import Tensor, nn
from torchaudio.models import Conformer


class VoiceEncoder(nn.Module):
    """StyleTTS-ZS joint acoustic and text voice encoder."""

    def __init__(
        self,
        mel_dim: int = 80,
        text_dim: int = 512,
        voice_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Conv1d(
            mel_dim,
            text_dim,
            kernel_size=3,
            padding=1,
        )
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
