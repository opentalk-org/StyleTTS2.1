import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedAttentivePool1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        attention_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, 1, 1),
        )
        self.output = nn.Linear(input_channels * 2, output_channels)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3 or mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("attentive pooling requires [B,C,T] and [B,1,T]")
        if torch.any(mask.sum(dim=2) == 0):
            raise ValueError("attentive pooling requires a valid token per item")
        logits = self.attention(features * mask).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=2)
        mean = (features * weights).sum(dim=2)
        variance = ((features - mean.unsqueeze(2)).square() * weights).sum(dim=2)
        statistics = torch.cat((mean, torch.sqrt(variance.clamp_min(1e-5))), dim=1)
        return self.output(statistics)


def pairwise_pool_tokens(tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 3 or mask.shape != (tokens.shape[0], 1, tokens.shape[2]):
        raise ValueError("pairwise pooling requires [B,C,T] and [B,1,T]")
    if tokens.shape[-1] % 2:
        tokens = F.pad(tokens, (0, 1))
        mask = F.pad(mask, (0, 1))
    batch, channels, frames = tokens.shape
    paired_tokens = tokens.view(batch, channels, frames // 2, 2)
    paired_mask = mask.view(batch, 1, frames // 2, 2)
    count = paired_mask.sum(dim=3).clamp_min(1)
    pooled = (paired_tokens * paired_mask).sum(dim=3) / count
    output_mask = paired_mask.any(dim=3)
    return pooled * output_mask, output_mask
