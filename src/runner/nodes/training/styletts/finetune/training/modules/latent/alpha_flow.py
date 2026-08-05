from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .denoiser import StyleDiffuser
from .prosody import STYLE_TOKEN_COUNT, STYLE_TOKEN_DIM


class AlphaFlow(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        style_scale: float = 1.0,
        transition_start: int = 1_000,
        transition_end: int = 4_000,
        temperature: float = 25.0,
        conditional_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.denoiser = StyleDiffuser(text_dim=text_dim)

        self.register_buffer("style_scale", torch.tensor(float(style_scale)))
        self.register_buffer("style_scale_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_raw_mse", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_velocity_cosine", torch.tensor(float("nan")), persistent=False)
        self.transition_start = transition_start
        self.transition_end = transition_end
        self.temperature = temperature
        self.conditional_dropout = conditional_dropout

    def schedule(self, step: int) -> float:
        if step <= self.transition_start:
            return 1.0
        if step >= self.transition_end:
            return 0.0
        scale = 1 / (self.transition_end - self.transition_start)
        offset = -(self.transition_start + self.transition_end) * scale / 2
        alpha = 1 - torch.sigmoid(torch.tensor((scale * step + offset) * self.temperature)).item()
        if alpha > 0.995:
            return 1.0
        if alpha < 0.005:
            return 0.0
        return alpha

    def forward(
        self,
        target: Tensor,
        embedding: Tensor,
        features: Tensor,
        input_lengths: Tensor,
        step: int,
    ) -> Tensor:
        scale = target.detach().float().flatten().std().clamp_min(1e-6)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scale, op=dist.ReduceOp.AVG)
        with torch.no_grad():
            self.style_scale_updates.add_(1)
            updates = self.style_scale_updates.to(scale.dtype)
            self.style_scale.add_((scale.to(self.style_scale) - self.style_scale) / updates)
        target = target / scale.to(target)
        batch = target.size(0)
        a = torch.sigmoid(
            torch.randn(batch, device=target.device, dtype=target.dtype) * 1.0 - 0.4
        )
        b = torch.sigmoid(
            torch.randn(batch, device=target.device, dtype=target.dtype) * 1.0 - 0.4
        )
        t = torch.maximum(a, b)
        r = torch.minimum(a, b)
        noise = torch.randn_like(target)
        velocity = noise - target
        noisy = (1 - t[:, None, None]) * target + t[:, None, None] * noise
        alpha = self.schedule(step)
        prediction = self.denoiser(
            noisy,
            r,
            t,
            input_lengths,
            embedding,
            features,
            embedding_mask_proba=self.conditional_dropout,
        )
        if alpha == 0.0:
            target_velocity = self._meanflow_target(
                noisy,
                r,
                t,
                velocity,
                embedding,
                features,
                input_lengths,
            )
        else:
            s = alpha * r + (1 - alpha) * t
            shifted = noisy - (t - s)[:, None, None] * velocity
            with torch.no_grad():
                shifted_velocity = self.denoiser(
                    shifted,
                    r,
                    s,
                    input_lengths,
                    embedding,
                    features,
                )
                target_velocity = alpha * velocity + (1 - alpha) * shifted_velocity
        error = prediction - target_velocity.detach()
        with torch.no_grad():
            self.last_raw_mse.copy_(error.float().square().mean())
            cosine = torch.nn.functional.cosine_similarity(
                prediction.float().flatten(1),
                target_velocity.detach().float().flatten(1),
                dim=1,
            ).mean()
            self.last_velocity_cosine.copy_(cosine)
        squared = error.flatten(1).square().sum(1)
        numerator = alpha if alpha > 0 else 1.0
        weight = numerator / (squared.detach() + 1e-3)
        return (weight * squared).mean()

    def _meanflow_target(
        self,
        noisy: Tensor,
        r: Tensor,
        t: Tensor,
        velocity: Tensor,
        embedding: Tensor,
        features: Tensor,
        input_lengths: Tensor,
    ) -> Tensor:
        evaluate = partial(
            self._velocity_at_end,
            r=r,
            embedding=embedding,
            features=features,
            input_lengths=input_lengths,
        )
        with sdpa_kernel(SDPBackend.MATH):
            _, derivative = torch.autograd.functional.jvp(
                evaluate,
                (noisy.detach().requires_grad_(True), t.detach().requires_grad_(True)),
                (velocity, torch.ones_like(t)),
                create_graph=False,
                strict=False,
            )
        return velocity - (t - r)[:, None, None] * derivative

    def _velocity_at_end(
        self,
        noisy: Tensor,
        t: Tensor,
        *,
        r: Tensor,
        embedding: Tensor,
        features: Tensor,
        input_lengths: Tensor,
    ) -> Tensor:
        return self.denoiser(noisy, r, t, input_lengths, embedding, features)

    def sample(
        self,
        embedding: Tensor,
        features: Tensor,
        input_lengths: Tensor,
        embedding_scale: float = 1.0,
        feature_scale: float = 1.0,
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn(
                embedding.size(0),
                STYLE_TOKEN_DIM,
                STYLE_TOKEN_COUNT,
                device=embedding.device,
                dtype=embedding.dtype,
            )
        r = torch.zeros(embedding.size(0), device=embedding.device, dtype=embedding.dtype)
        t = torch.ones_like(r)
        velocity = self.denoiser(
            noise,
            r,
            t,
            input_lengths,
            embedding,
            features,
            embedding_scale=embedding_scale,
            feature_scale=feature_scale,
        )
        return (noise - velocity) * self.style_scale.to(noise)
