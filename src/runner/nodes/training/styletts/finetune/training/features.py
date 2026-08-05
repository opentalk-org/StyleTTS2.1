import torch
import torchaudio
from torch import Tensor, nn
from transformers import (
    AutoModel,
    WavLMForXVector,
)

SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"


class WavLMFeatures(nn.Module):
    def __init__(self, model: str, model_sr: int, slm_sr: int) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model).requires_grad_(False)
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, waveform: Tensor):
        waveform = self.resample(waveform.float())
        return self.model(
            input_values=waveform,
            output_hidden_states=True,
        ).hidden_states

    @staticmethod
    def discriminator_input(features) -> Tensor:
        return torch.stack(features, dim=1).transpose(-1, -2).flatten(1, 2)


class SpeakerFeatures(nn.Module):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.resample = torchaudio.transforms.Resample(sample_rate, 16_000)
        self.model = (
            WavLMForXVector.from_pretrained(SPEAKER_MODEL)
            .requires_grad_(False)
            .eval()
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        waveform = waveform.squeeze(1) if waveform.dim() == 4 else waveform
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        waveform = self.resample(waveform.squeeze(1))
        input_values = waveform.float()
        input_values = (input_values - input_values.mean(-1, keepdim=True)) / (
            input_values.var(-1, keepdim=True, unbiased=False) + 1e-7
        ).sqrt()
        embedding = self.model(input_values=input_values).embeddings
        return input_values, embedding
