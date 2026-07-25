from pathlib import Path

import soundfile as sf
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import functional as F

from .types import SignalPair, ValidationArtifactSet


_COMMON_NAMES = (
    "f0.png",
    "gt.wav",
    "latent.png",
    "mel.png",
    "n.png",
    "phase.png",
    "stft_magnitude.png",
)


def artifact_names() -> tuple[str, ...]:
    return tuple(sorted((*_COMMON_NAMES, "pred.wav", "alignment.png")))


def render_validation_sample(
    sample: ValidationArtifactSet,
    directory: Path,
    sample_rate: int,
) -> tuple[Path, ...]:
    _validate_cpu_sample(sample)
    directory.mkdir(parents=True, exist_ok=True)
    _write_wave(directory / "gt.wav", sample.ground_truth, sample_rate)
    _write_wave(directory / "pred.wav", sample.prediction, sample_rate)
    _matrix_plot(directory / "latent.png", sample.latent, "Latent")
    _line_pair(directory / "f0.png", sample.f0, "F0")
    _line_pair(directory / "n.png", sample.n, "N")
    _matrix_pair(directory / "mel.png", sample.mel, "Mel")
    magnitude, phase = _spectrogram_pairs(sample.ground_truth, sample.prediction)
    _matrix_pair(directory / "stft_magnitude.png", magnitude, "STFT magnitude")
    _matrix_pair(directory / "phase.png", phase, "STFT phase")
    _matrix_plot(directory / "alignment.png", sample.alignment, "Alignment")
    return tuple(directory / name for name in artifact_names())


def _validate_cpu_sample(sample: ValidationArtifactSet) -> None:
    tensors = (
        sample.ground_truth,
        sample.prediction,
        sample.latent,
        *sample.f0,
        *sample.n,
        *sample.mel,
        *((sample.alignment,) if sample.alignment is not None else ()),
    )
def _write_wave(path: Path, waveform: Tensor, sample_rate: int) -> None:
    values = waveform.float().squeeze(0).clamp(-1, 1).numpy()
    sf.write(path, values, sample_rate, subtype="PCM_16")


def _line_pair(path: Path, pair: SignalPair, title: str) -> None:
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    axis.plot(pair[0].float().flatten().numpy(), label="ground truth")
    axis.plot(pair[1].float().flatten().numpy(), label="prediction")
    axis.set_title(title)
    axis.legend()
    figure.savefig(path, dpi=120)


def _matrix_plot(path: Path, values: Tensor, title: str) -> None:
    matrix = values.float().squeeze().numpy()
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    figure = Figure(figsize=(10, 4), layout="constrained")
    axis = figure.subplots()
    axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_title(title)
    figure.savefig(path, dpi=120)


def _matrix_pair(path: Path, pair: SignalPair, title: str) -> None:
    figure = Figure(figsize=(10, 7), layout="constrained")
    axes = figure.subplots(2, 1)
    for axis, values, label in zip(
        axes,
        pair,
        ("ground truth", "prediction"),
        strict=True,
    ):
        matrix = values.float().squeeze().numpy()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        axis.imshow(matrix, origin="lower", aspect="auto")
        axis.set_title(f"{title}: {label}")
    figure.savefig(path, dpi=120)


def _spectrogram_pairs(target: Tensor, prediction: Tensor) -> tuple[SignalPair, SignalPair]:
    values = []
    for waveform in (target, prediction):
        signal = waveform.float().flatten()
        padded = F.pad(signal, (0, max(0, 512 - signal.numel())))
        spectrum = torch.stft(
            padded,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window=torch.hann_window(512),
            return_complex=True,
        )
        values.append((torch.log1p(spectrum.abs()), torch.angle(spectrum)))
    return (values[0][0], values[1][0]), (values[0][1], values[1][1])
