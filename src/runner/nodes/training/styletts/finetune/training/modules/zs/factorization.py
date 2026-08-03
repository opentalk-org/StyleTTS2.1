import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradients: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradients, None


class SameSpeakerVerifier(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left = left[:, None, :]
        right = right[None, :, :]
        relation = torch.cat(
            (
                (left - right).abs(),
                left * right,
                F.cosine_similarity(left, right, dim=-1).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.classifier(relation).squeeze(-1)


class FactorizationHeads(nn.Module):
    def __init__(
        self,
        language_count: int,
        content_dim: int,
        voice_dim: int = 128,
        style_dim: int = 512,
    ) -> None:
        super().__init__()
        self.style_verifier = SameSpeakerVerifier(voice_dim)
        self.style_language = nn.Linear(voice_dim, language_count)
        self.style_content = nn.Linear(voice_dim, content_dim)
        self.style_projection = nn.Linear(style_dim, voice_dim)

    def style_nuisance_loss(
        self,
        style: Tensor,
        speaker_ids: Tensor,
        language_ids: Tensor,
        content_bag: Tensor,
        reversal_strength: float,
    ) -> Tensor:
        style = style.mean(-1)
        reversed_style = GradientReversal.apply(style, reversal_strength)
        style_features = self.style_projection(reversed_style)
        style_language = F.cross_entropy(self.style_language(style_features), language_ids)
        style_content = F.binary_cross_entropy_with_logits(
            self.style_content(style_features),
            content_bag,
        )
        speaker_logits = self.style_verifier(style_features, style_features)
        same = speaker_ids[:, None].eq(speaker_ids[None, :]).to(speaker_logits.dtype)
        style_speaker = F.binary_cross_entropy_with_logits(speaker_logits, same)
        return style_language + style_speaker + style_content

    def cross_covariance(self, voice: Tensor, style: Tensor) -> Tensor:
        if voice.size(0) < 2:
            return voice.new_zeros(())
        style = F.linear(
            style.mean(-1),
            self.style_projection.weight.detach(),
            self.style_projection.bias.detach(),
        )
        voice = (voice - voice.mean(0)) / voice.std(0, unbiased=False).clamp_min(1e-4)
        style = (style - style.mean(0)) / style.std(0, unbiased=False).clamp_min(1e-4)
        return (voice.transpose(0, 1) @ style / (voice.size(0) - 1)).square().mean()
