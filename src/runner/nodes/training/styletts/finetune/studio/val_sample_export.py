from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ValidationSample:
    ground_truth: Tensor
    prediction: Tensor
    target_f0: Tensor
    predicted_f0: Tensor
    target_n: Tensor
    predicted_n: Tensor
    soft_attention: Tensor
    hard_attention: Tensor


@dataclass(frozen=True)
class ValidationSampleArtifacts:
    index: int
    paths: tuple[Path, ...]


class ValidationArtifactRenderer:
    def __init__(self, output_path: Path, sample_rate: int) -> None:
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 300
        self.win_length = 1200
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=80,
        )

    def render(
        self,
        step: int,
        samples: list[ValidationSample],
    ) -> list[ValidationSampleArtifacts]:
        artifacts = []
        for index, sample in enumerate(samples):
            relative = Path("samples", f"step_{step:09d}", f"sample_{index}")
            directory = self.output_path / relative
            directory.mkdir(parents=True, exist_ok=True)
            paths = (
                self._wave(directory / "gt.wav", sample.ground_truth),
                self._wave(directory / "pred.wav", sample.prediction),
                self._lines(
                    directory / "f0.png",
                    sample.target_f0,
                    sample.predicted_f0,
                    "F0",
                ),
                self._lines(
                    directory / "n.png",
                    sample.target_n,
                    sample.predicted_n,
                    "N",
                ),
                self._matrix(
                    directory / "soft_attention.png",
                    sample.soft_attention,
                    "Soft attention",
                ),
                self._matrix(
                    directory / "hard_attention.png",
                    sample.hard_attention,
                    "Hard attention",
                ),
                self._matrix_pair(
                    directory / "mel.png",
                    self._mel(sample.ground_truth),
                    self._mel(sample.prediction),
                    "Mel",
                ),
                self._matrix_pair(
                    directory / "stft.png",
                    self._stft(sample.ground_truth).abs().log1p(),
                    self._stft(sample.prediction).abs().log1p(),
                    "STFT magnitude",
                ),
                self._matrix_pair(
                    directory / "phase.png",
                    torch.angle(self._stft(sample.ground_truth)),
                    torch.angle(self._stft(sample.prediction)),
                    "STFT phase",
                ),
            )
            artifacts.append(
                ValidationSampleArtifacts(
                    index=index,
                    paths=tuple(path.relative_to(self.output_path) for path in paths),
                )
            )
        return artifacts

    def _wave(self, path: Path, waveform: Tensor) -> Path:
        values = waveform.detach().float().cpu().flatten().clamp(-1, 1).numpy()
        sf.write(path, values, self.sample_rate, subtype="PCM_16")
        return path

    def _mel(self, waveform: Tensor) -> Tensor:
        signal = waveform.detach().float().cpu().flatten()
        return torch.log(self.mel_transform(signal).clamp_min(1e-5))

    def _stft(self, waveform: Tensor) -> Tensor:
        signal = waveform.detach().float().cpu().flatten()
        signal = F.pad(signal, (0, max(0, self.n_fft - signal.numel())))
        return torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            return_complex=True,
        )

    @staticmethod
    def _lines(
        path: Path,
        target: Tensor,
        prediction: Tensor,
        title: str,
    ) -> Path:
        figure = Figure(figsize=(10, 5), layout="constrained")
        axis = figure.subplots()
        axis.plot(target.detach().float().cpu().flatten().numpy(), label="ground truth")
        axis.plot(prediction.detach().float().cpu().flatten().numpy(), label="prediction")
        axis.set_title(title)
        axis.legend()
        figure.savefig(path, dpi=120)
        return path

    @staticmethod
    def _matrix(path: Path, values: Tensor, title: str) -> Path:
        matrix = values.detach().float().cpu().squeeze().numpy()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        figure = Figure(figsize=(10, 4), layout="constrained")
        axis = figure.subplots()
        axis.imshow(matrix, origin="lower", aspect="auto")
        axis.set_title(title)
        figure.savefig(path, dpi=120)
        return path

    @staticmethod
    def _matrix_pair(
        path: Path,
        target: Tensor,
        prediction: Tensor,
        title: str,
    ) -> Path:
        figure = Figure(figsize=(10, 7), layout="constrained")
        axes = figure.subplots(2, 1)
        for axis, values, label in zip(
            axes,
            (target, prediction),
            ("ground truth", "prediction"),
            strict=True,
        ):
            matrix = values.detach().float().cpu().squeeze().numpy()
            if matrix.ndim == 1:
                matrix = matrix[None, :]
            axis.imshow(matrix, origin="lower", aspect="auto")
            axis.set_title(f"{title}: {label}")
        figure.savefig(path, dpi=120)
        return path
