import numpy as np
import torch


class SineGenerator(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        upsample_scale,
        harmonic_count=0,
        sine_amplitude=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        pulse=False,
    ):
        super().__init__()
        self.sine_amplitude = sine_amplitude
        self.noise_std = noise_std
        self.dim = harmonic_count + 1
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.pulse = pulse
        self.upsample_scale = upsample_scale

    def _voiced(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _sine(self, f0_values):
        phase_steps = (f0_values / self.sample_rate) % 1
        initial_phase = torch.rand(
            f0_values.shape[0],
            f0_values.shape[2],
            device=f0_values.device,
        )
        initial_phase[:, 0] = 0
        phase_steps[:, 0, :] += initial_phase

        if not self.pulse:
            phase_steps = torch.nn.functional.interpolate(
                phase_steps.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            phase = torch.cumsum(phase_steps, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            return torch.sin(phase)

        voiced = self._voiced(f0_values)
        next_voiced = torch.roll(voiced, shifts=-1, dims=1)
        next_voiced[:, -1, :] = 1
        starts = (voiced < 1) * (next_voiced > 0)
        cumulative = torch.cumsum(phase_steps, dim=1)
        for index in range(f0_values.shape[0]):
            cycle_sums = cumulative[index, starts[index, :, 0], :]
            cycle_sums[1:, :] = cycle_sums[1:, :] - cycle_sums[:-1, :]
            cumulative[index, :, :] = 0
            cumulative[index, starts[index, :, 0], :] = cycle_sums
        phase = torch.cumsum(phase_steps - cumulative, dim=1)
        return torch.cos(phase * 2 * np.pi)

    def forward(self, f0):
        harmonics = torch.arange(1, self.dim + 1, device=f0.device)
        frequencies = f0 * harmonics
        sine_waves = self._sine(frequencies) * self.sine_amplitude
        voiced = self._voiced(f0)
        noise_amplitude = (
            voiced * self.noise_std
            + (1 - voiced) * self.sine_amplitude / 3
        )
        noise = noise_amplitude * torch.randn_like(sine_waves)
        return sine_waves * voiced + noise, voiced, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGenerator(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0):
        with torch.no_grad():
            sine_waves, voiced, _ = self.l_sin_gen(f0)
        sine_merge = self.l_tanh(self.l_linear(sine_waves))
        noise = torch.randn_like(voiced) * self.sine_amp / 3
        return sine_merge, noise, voiced
