import torch
import torch.nn.functional as F
from pyannote.audio import Model
from torch import Tensor, nn
from torchaudio.transforms import Resample


SPEAKER_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


class _FeatureCapture:
    def __init__(self, layers: list[nn.Module]) -> None:
        self.features: list[Tensor] = []
        self.handles = [layer.register_forward_hook(self) for layer in layers]

    def __call__(self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.features.append(output)

    def reset(self) -> None:
        self.features.clear()


class SpeakerVerificationLoss(nn.Module):
    def __init__(self, model: Model, sample_rate: int) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.resample = Resample(sample_rate, 16_000)
        self.capture = _FeatureCapture([
            self.model.resnet.layer1,
            self.model.resnet.layer2,
            self.model.resnet.layer3,
            self.model.resnet.layer4,
        ])

    def _encode(self, waveform: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        self.capture.reset()
        embedding = self.model(self.resample(waveform.float()))
        return tuple(self.capture.features), embedding

    def forward(self, real: Tensor, generated: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            real_features, real_embedding = self._encode(real)
        generated_features, generated_embedding = self._encode(generated)
        feature = generated.new_zeros((), dtype=torch.float32)
        for generated_hidden, real_hidden in zip(
            generated_features,
            real_features,
            strict=True,
        ):
            feature = feature + F.l1_loss(generated_hidden, real_hidden)
        similarity = 1 - F.cosine_similarity(
            generated_embedding,
            real_embedding,
            dim=-1,
        ).mean()
        return feature, similarity


def load_speaker_verification_loss(sample_rate: int, device: torch.device) -> SpeakerVerificationLoss:
    model = Model.from_pretrained(SPEAKER_MODEL)
    assert model is not None, f"failed to load {SPEAKER_MODEL}"
    return SpeakerVerificationLoss(model, sample_rate).to(device)
