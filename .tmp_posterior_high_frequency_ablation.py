from pathlib import Path

import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 24_000
N_FFT = 2_048
HOP_LENGTH = 300
WIN_LENGTH = 1_200
CUTOFFS_HZ = (1_000, 2_000, 4_000, 6_000, 8_000)
SAMPLES = (8, 16)
RUN = Path(
    "src/runner/nodes/training/beetle/runs/"
    "libritts-train-clean-100-logspecdisc-10k-v2-20260724"
)


def spectrum(waveform: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )


def waveform(spectrum_value: torch.Tensor, window: torch.Tensor, length: int) -> torch.Tensor:
    return torch.istft(
        spectrum_value,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=length,
    )


def write(path: Path, values: torch.Tensor) -> None:
    sf.write(path, values.clamp(-1, 1).numpy(), SAMPLE_RATE, subtype="PCM_16")


def render_sample(sample_index: int) -> tuple[float, float]:
    source = RUN / "validation/training/step_3000/audio" / f"sample_{sample_index}"
    output = (
        RUN
        / "analysis/step_3000/audio"
        / f"sample_{sample_index}/high_frequency_ablation"
    )
    ground_truth, ground_truth_rate = torchaudio.load(source / "gt.wav")
    prediction, prediction_rate = torchaudio.load(source / "pred.wav")
    if ground_truth_rate != SAMPLE_RATE or prediction_rate != SAMPLE_RATE:
        raise ValueError("ablation inputs must use the configured sample rate")
    length = min(ground_truth.shape[-1], prediction.shape[-1])
    ground_truth = ground_truth[0, :length]
    prediction = prediction[0, :length]
    window = torch.hann_window(WIN_LENGTH)
    ground_truth_spectrum = spectrum(ground_truth, window)
    prediction_spectrum = spectrum(prediction, window)
    frequencies = torch.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)

    output.mkdir(parents=True, exist_ok=True)
    write(output / "00_prediction_original.wav", prediction)
    prediction_roundtrip = waveform(prediction_spectrum, window, length)
    write(output / "01_prediction_stft_roundtrip.wav", prediction_roundtrip)
    for position, cutoff_hz in enumerate(CUTOFFS_HZ, start=2):
        use_ground_truth = frequencies[:, None] >= cutoff_hz
        hybrid = torch.where(
            use_ground_truth,
            ground_truth_spectrum,
            prediction_spectrum,
        )
        write(
            output
            / f"{position:02d}_prediction_below_{cutoff_hz // 1000}khz_gt_above.wav",
            waveform(hybrid, window, length),
        )

    use_ground_truth = frequencies[:, None] >= 4_000
    magnitude_hybrid = torch.polar(
        torch.where(
            use_ground_truth,
            ground_truth_spectrum.abs(),
            prediction_spectrum.abs(),
        ),
        torch.angle(prediction_spectrum),
    )
    phase_hybrid = torch.polar(
        prediction_spectrum.abs(),
        torch.where(
            use_ground_truth,
            torch.angle(ground_truth_spectrum),
            torch.angle(prediction_spectrum),
        ),
    )
    inverse_hybrid = torch.where(
        use_ground_truth,
        prediction_spectrum,
        ground_truth_spectrum,
    )
    write(
        output / "07_prediction_with_gt_magnitude_above_4khz.wav",
        waveform(magnitude_hybrid, window, length),
    )
    write(
        output / "08_prediction_with_gt_phase_above_4khz.wav",
        waveform(phase_hybrid, window, length),
    )
    write(
        output / "09_gt_below_4khz_prediction_above.wav",
        waveform(inverse_hybrid, window, length),
    )
    write(
        output / "10_ground_truth_stft_roundtrip.wav",
        waveform(ground_truth_spectrum, window, length),
    )
    error = prediction - prediction_roundtrip
    return float(error.square().mean().sqrt()), float(error.abs().max())


def main() -> None:
    for sample_index in SAMPLES:
        rmse, peak = render_sample(sample_index)
        print(sample_index, "roundtrip_rmse", rmse, "roundtrip_peak_error", peak)


if __name__ == "__main__":
    main()
