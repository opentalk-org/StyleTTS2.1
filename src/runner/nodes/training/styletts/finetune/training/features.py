import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torchaudio.compliance import kaldi
from transformers import (
    AutoModel,
)

from .modules.tidyvoice import TidyVoiceSpeakerModel

SPEAKER_MODEL = "areffarhadi/Resnet34-tidyvoiceX-ASV"


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
        checkpoint = hf_hub_download(SPEAKER_MODEL, "models/avg_model.pt")
        self.model = TidyVoiceSpeakerModel()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        del state["projection.weight"]
        self.model.load_state_dict(
            state,
            strict=True,
        )
        self.model.requires_grad_(False).eval()

    @staticmethod
    def _fbank(waveform: Tensor) -> Tensor:
        values = kaldi.fbank(
            waveform * (1 << 15),
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0,
            sample_frequency=16_000,
            window_type="hamming",
            use_energy=False,
        )
        return values - values.mean(dim=0, keepdim=True)

    def forward(self, waveform: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        waveform = waveform.squeeze(1) if waveform.dim() == 4 else waveform
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        waveform = self.resample(waveform).float()
        fbank = torch.stack([self._fbank(item) for item in waveform])
        return self.model(fbank)
