import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .config.architecture import AudioConfig
from .losses.acoustic import LogMelSpectrogram
from .mlflow_logging import MlflowLogger


@dataclass(frozen=True)
class ValidationSample:
    ground_truth: Tensor
    prediction: Tensor | None
    target_latent: Tensor
    predicted_latent: Tensor | None
    target_f0: Tensor | None
    predicted_f0: Tensor | None
    target_n: Tensor | None
    predicted_n: Tensor | None
    alignment: Tensor | None
    soft_alignment: Tensor | None


class ValidationArtifacts:
    def __init__(
        self,
        output_path: Path,
        audio: AudioConfig,
        logger: MlflowLogger,
    ) -> None:
        self.output_path = output_path
        self.sample_rate = audio.sample_rate
        self.n_fft = audio.n_fft
        self.hop_length = audio.hop_length
        self.win_length = audio.win_length
        self.mel_transform = LogMelSpectrogram(
            audio.sample_rate,
            audio.n_fft,
            audio.hop_length,
            audio.win_length,
            audio.mel_channels,
            audio.f_min,
            audio.f_max,
        )
        self.logger = logger

    def publish(
        self,
        step: int,
        position: int,
        sample: ValidationSample,
        metrics: dict[str, float],
    ) -> None:
        relative = Path(
            "validation",
            "training",
            f"step_{step}",
            f"sample_{position}",
        )
        directory = self.output_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        ground_truth = sample.ground_truth.detach().float().cpu()
        target_wave = directory / "gt.wav"
        sf.write(
            target_wave,
            ground_truth.squeeze().clamp(-1, 1).numpy(),
            self.sample_rate,
            subtype="PCM_16",
        )
        paths.append(target_wave)
        if sample.prediction is not None:
            prediction = sample.prediction.detach().float().cpu()
            predicted_wave = directory / "pred.wav"
            sf.write(
                predicted_wave,
                prediction.squeeze().clamp(-1, 1).numpy(),
                self.sample_rate,
                subtype="PCM_16",
            )
            paths.append(predicted_wave)
            paths.append(
                self._matrix_pair(
                    directory / "mel.png",
                    self._mel(ground_truth),
                    self._mel(prediction),
                    "Mel",
                )
            )
            paths.append(
                self._matrix_pair(
                    directory / "stft.png",
                    self._stft(ground_truth),
                    self._stft(prediction),
                    "STFT magnitude",
                )
            )
            paths.append(
                self._matrix_pair(
                    directory / "phase.png",
                    self._phase(ground_truth),
                    self._phase(prediction),
                    "STFT phase (radians)",
                )
            )
        if sample.predicted_latent is None:
            paths.append(
                self._latent(
                    directory / "latent.png",
                    sample.target_latent,
                    "Target latent",
                )
            )
        else:
            paths.append(
                self._latent_pair(
                    directory / "latent.png",
                    sample.target_latent,
                    sample.predicted_latent,
                    "Latent",
                )
            )
        if sample.target_f0 is not None:
            paths.append(
                self._lines(
                    directory / "f0.png",
                    sample.target_f0,
                    sample.predicted_f0,
                    "F0",
                )
            )
        if sample.target_n is not None:
            paths.append(
                self._lines(
                    directory / "n.png",
                    sample.target_n,
                    sample.predicted_n,
                    "N",
                )
            )
        if sample.alignment is not None:
            paths.append(
                self._matrix(
                    directory / "alignment.png",
                    sample.alignment,
                    "Hard phoneme alignment",
                )
            )
        if sample.soft_alignment is not None:
            paths.append(
                self._matrix(
                    directory / "soft_alignment.png",
                    sample.soft_alignment,
                    "Soft phoneme alignment",
                )
            )
        manifest = directory / "metrics.json"
        manifest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        paths.append(manifest)
        for path in paths:
            self.logger.log_artifact(path, str(relative))

    def _mel(self, waveform: Tensor) -> Tensor:
        return self.mel_transform(waveform.reshape(1, -1)).squeeze(0)

    def _stft(self, waveform: Tensor) -> Tensor:
        signal = waveform.flatten()
        padding = (self.n_fft - self.hop_length) // 2
        padded = torch.nn.functional.pad(
            signal.reshape(1, 1, -1),
            (padding, padding),
            mode="reflect",
        ).reshape(-1)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=False,
            return_complex=True,
        )
        return torch.log(torch.clamp(spectrum.abs(), min=1e-5))

    def _phase(self, waveform: Tensor) -> Tensor:
        signal = waveform.flatten()
        padding = (self.n_fft - self.hop_length) // 2
        padded = torch.nn.functional.pad(
            signal.reshape(1, 1, -1),
            (padding, padding),
            mode="reflect",
        ).reshape(-1)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=False,
            return_complex=True,
        )
        return torch.angle(spectrum)

    def _matrix(self, path: Path, values: Tensor, title: str) -> Path:
        matrix = values.detach().float().cpu().squeeze().numpy()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        figure = Figure(figsize=(10, 4), layout="constrained")
        axis = figure.subplots()
        axis.imshow(matrix, origin="lower", aspect="auto")
        axis.set_title(title)
        figure.savefig(path, dpi=120)
        return path

    def _latent(self, path: Path, values: Tensor, title: str) -> Path:
        matrix = values.detach().float().cpu().squeeze().numpy()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        deviation = matrix.std(axis=1, keepdims=True)
        figure = Figure(figsize=(11, 4), layout="constrained")
        matrix_axis, deviation_axis = figure.subplots(
            1,
            2,
            gridspec_kw={"width_ratios": (20, 1)},
        )
        matrix_axis.imshow(matrix, origin="lower", aspect="auto")
        matrix_axis.set_title(title)
        deviation_axis.imshow(deviation, origin="lower", aspect="auto")
        deviation_axis.set_title("Std")
        deviation_axis.set_xticks(())
        deviation_axis.set_yticks(())
        figure.savefig(path, dpi=120)
        return path

    def _latent_pair(
        self,
        path: Path,
        target: Tensor,
        prediction: Tensor,
        title: str,
    ) -> Path:
        matrices = (
            target.detach().float().cpu().squeeze().numpy(),
            prediction.detach().float().cpu().squeeze().numpy(),
        )
        matrices = tuple(
            matrix[None, :] if matrix.ndim == 1 else matrix
            for matrix in matrices
        )
        deviations = tuple(
            matrix.std(axis=1, keepdims=True) for matrix in matrices
        )
        maximum_deviation = max(float(values.max()) for values in deviations)
        figure = Figure(figsize=(11, 7), layout="constrained")
        axes = figure.subplots(
            2,
            2,
            gridspec_kw={"width_ratios": (20, 1)},
        )
        for row, (matrix, deviation, label) in enumerate(
            zip(
                matrices,
                deviations,
                ("ground truth", "prediction"),
                strict=True,
            )
        ):
            axes[row, 0].imshow(matrix, origin="lower", aspect="auto")
            axes[row, 0].set_title(f"{title}: {label}")
            axes[row, 1].imshow(
                deviation,
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=maximum_deviation,
            )
            axes[row, 1].set_title("Std")
            axes[row, 1].set_xticks(())
            axes[row, 1].set_yticks(())
        figure.savefig(path, dpi=120)
        return path

    def _matrix_pair(
        self,
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
            axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                resample=False,
            )
            axis.set_title(f"{title}: {label}")
        figure.savefig(path, dpi=120)
        return path

    def _lines(
        self,
        path: Path,
        target: Tensor,
        prediction: Tensor | None,
        title: str,
    ) -> Path:
        figure = Figure(figsize=(10, 5), layout="constrained")
        axis = figure.subplots()
        axis.plot(target.detach().float().cpu().flatten().numpy(), label="ground truth")
        if prediction is not None:
            axis.plot(
                prediction.detach().float().cpu().flatten().numpy(),
                label="prediction",
            )
        axis.set_title(title)
        axis.legend()
        figure.savefig(path, dpi=120)
        return path
