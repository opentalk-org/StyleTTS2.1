import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PQMF(nn.Module):
    def __init__(
        self,
        subbands: int,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ) -> None:
        super().__init__()
        if taps % 2:
            raise ValueError("PQMF taps must be even")
        positions = torch.arange(taps + 1, dtype=torch.float64) - taps / 2
        prototype = cutoff_ratio * torch.sinc(cutoff_ratio * positions)
        prototype *= torch.kaiser_window(
            taps + 1,
            periodic=False,
            beta=beta,
            dtype=torch.float64,
        )
        filters = []
        for band in range(subbands):
            phase = (
                (2 * band + 1) * torch.pi / (2 * subbands) * positions
                - ((-1) ** band) * torch.pi / 4
            )
            filters.append(2 * prototype * torch.cos(phase))
        synthesis = torch.stack(filters).float().unsqueeze(0)
        upsample = torch.zeros(subbands, subbands, subbands)
        indices = torch.arange(subbands)
        upsample[indices, indices, 0] = 1
        self.register_buffer("synthesis_filter", synthesis)
        self.register_buffer("upsample_filter", upsample)
        self.padding = nn.ConstantPad1d(taps // 2, 0)
        self.subbands = subbands

    def synthesize(self, subband_waveforms: Tensor) -> Tensor:
        upsampled = F.conv_transpose1d(
            subband_waveforms,
            self.upsample_filter * self.subbands,
            stride=self.subbands,
        )
        return F.conv1d(self.padding(upsampled), self.synthesis_filter)


class MultiBandISTFT(nn.Module):
    def __init__(self, subbands: int, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.subbands = subbands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pqmf = PQMF(subbands)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, spectrogram: Tensor) -> Tensor:
        batch, channels, frequency_bins, frames = spectrogram.shape
        if channels != self.subbands * 2:
            raise ValueError("iSTFT spectrogram must contain magnitude/phase per subband")
        if frequency_bins != self.n_fft // 2 + 1:
            raise ValueError("iSTFT spectrogram frequency-bin count is invalid")
        bands = spectrogram.view(
            batch,
            self.subbands,
            2,
            frequency_bins,
            frames,
        )
        band_length = frames * self.hop_length
        waveforms = []
        for band in range(self.subbands):
            magnitude = torch.exp(bands[:, band, 0].float())
            phase = torch.sin(bands[:, band, 1].float())
            spectrum = torch.polar(magnitude, phase)
            waveforms.append(
                torch.istft(
                    spectrum,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=self.window,
                    center=True,
                    length=band_length,
                )
            )
        waveform = self.pqmf.synthesize(torch.stack(waveforms, dim=1))
        expected = frames * self.hop_length * self.subbands
        if waveform.shape != (batch, 1, expected):
            raise ValueError(
                f"multiband synthesis produced {waveform.shape}; expected {(batch, 1, expected)}"
            )
        return waveform
