from functools import partial
from math import sqrt
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor

from .utils import default, exists


class Distribution:
    def __call__(self, num_samples: int, device: torch.device) -> Tensor:
        raise NotImplementedError


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self,
        num_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0) -> Tensor:
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    x_flat = rearrange(x, "b ... -> b (...)")
    scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
    scale.clamp_(min=1.0)
    scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
    return x.clamp(-scale, scale) / scale


def to_batch(
    batch_size: int,
    device: torch.device,
    x: float | None = None,
    xs: Tensor | None = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "exactly one of x or xs must be provided"
    if x is not None:
        return torch.full((batch_size,), x, device=device)
    assert xs is not None
    return xs


class Diffusion(nn.Module):
    alias = ""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        noise: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError


class KDiffusion(Diffusion):
    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,
        dynamic_threshold: float = 0.0,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = sigma_data**2 / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data * (sigma_data**2 + sigmas**2) ** -0.5
        c_in = (sigmas**2 + sigma_data**2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigma_batch = to_batch(
            x=sigma,
            xs=sigmas,
            batch_size=batch_size,
            device=device,
        )
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigma_batch)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        return c_skip * x_noisy + c_out * x_pred

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        return (sigmas**2 + self.sigma_data**2) * (
            sigmas * self.sigma_data
        ) ** -2

    def forward(
        self,
        x: Tensor,
        noise: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(batch_size, device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")
        resolved_noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * resolved_noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        return (losses * self.loss_weight(sigmas)).mean()


class Schedule(nn.Module):
    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError


class KarrasSchedule(Schedule):
    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return F.pad(sigmas, pad=(0, 1), value=0.0)


class Sampler(nn.Module):
    diffusion_types: tuple[type[Diffusion], ...] = ()

    def forward(
        self,
        noise: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
    ) -> Tensor:
        raise NotImplementedError


class ADPM2Sampler(Sampler):
    diffusion_types = (KDiffusion,)

    def __init__(self, rho: float = 1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(
        self,
        sigma: float,
        sigma_next: float,
    ) -> tuple[float, float, float]:
        sigma_up = sqrt(
            sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2
        )
        sigma_down = sqrt(sigma_next**2 - sigma_up**2)
        sigma_mid = (
            (sigma ** (1 / self.rho) + sigma_down ** (1 / self.rho)) / 2
        ) ** self.rho
        return sigma_up, sigma_down, sigma_mid

    def step(
        self,
        x: Tensor,
        fn: Callable,
        sigma: float,
        sigma_next: float,
    ) -> Tensor:
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        derivative = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + derivative * (sigma_mid - sigma)
        midpoint_derivative = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        x = x + midpoint_derivative * (sigma_down - sigma)
        return x + torch.randn_like(x) * sigma_up

    def forward(
        self,
        noise: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
    ) -> Tensor:
        x = sigmas[0] * noise
        for index in range(num_steps - 1):
            x = self.step(x, fn, sigmas[index], sigmas[index + 1])
        return x


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: int | None = None,
        clamp: bool = True,
    ):
        super().__init__()
        if diffusion.alias not in {
            diffusion_type.alias for diffusion_type in sampler.diffusion_types
        }:
            raise ValueError(
                f"{sampler.__class__.__name__} is incompatible with "
                f"{diffusion.__class__.__name__}"
            )
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

    def forward(
        self,
        noise: Tensor,
        num_steps: int | None = None,
        **kwargs,
    ) -> Tensor:
        resolved_steps = num_steps if num_steps is not None else self.num_steps
        if resolved_steps is None:
            raise ValueError("num_steps is required")
        sigmas = self.sigma_schedule(resolved_steps, noise.device)
        output = self.sampler(
            noise,
            fn=partial(self._denoise, call_kwargs=kwargs),
            sigmas=sigmas,
            num_steps=resolved_steps,
        )
        return output.clamp(-1.0, 1.0) if self.clamp else output

    def _denoise(self, *args, call_kwargs, **kwargs):
        return self.denoise_fn(*args, **kwargs, **call_kwargs)
