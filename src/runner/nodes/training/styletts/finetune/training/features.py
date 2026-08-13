from typing import Any, cast

import torch
import torch.nn.functional as F
import torchaudio
from pyannote.audio import Model
from torch import Tensor, nn
from transformers import (
    AutoModel,
)

SPEAKER_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


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
        model = Model.from_pretrained(SPEAKER_MODEL)
        if model is None:
            raise RuntimeError(f"could not load speaker model {SPEAKER_MODEL}")
        self.model = cast(Any, model.requires_grad_(False).eval())

    def forward(self, waveform: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        waveform = waveform.squeeze(1) if waveform.dim() == 4 else waveform
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        waveform = self.resample(waveform).float()
        fbank = self.model.compute_fbank(waveform)
        resnet = self.model.resnet
        hidden = fbank.permute(0, 2, 1).unsqueeze(1)
        hidden = F.relu(resnet.bn1(resnet.conv1(hidden)))
        features = []
        for stage in (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ):
            for block in stage:
                hidden = block(hidden)
                features.append(hidden)
        embedding = resnet.seg_1(resnet.pool(hidden))
        if resnet.two_emb_layer:
            embedding = resnet.seg_2(resnet.seg_bn_1(F.relu(embedding)))
        return tuple(features), embedding
