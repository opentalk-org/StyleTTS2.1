import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...losses import discriminator_TPRLS_loss, generator_TPRLS_loss


class ProsodyGeneratorLoss(nn.Module):
    """Generator and feature-matching objective from the author script."""

    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        predicted: Tensor,
        real: Tensor,
        style: Tensor,
        lengths: Tensor,
        max_size: int,
    ) -> tuple[Tensor, Tensor]:
        fake_scores, fake_features = self.discriminator(
            predicted,
            style.detach(),
            lengths,
            max_size,
        )
        with torch.no_grad():
            real_scores, real_features = self.discriminator(
                real.detach(),
                style.detach(),
                lengths,
                max_size,
            )
        adversarial = []
        feature_matching = []
        for index, length_value in enumerate(lengths):
            length = int(length_value.item())
            fake = fake_scores[index, :length]
            real_item = real_scores[index, :length]
            item = torch.mean((1 - fake).square()) + generator_TPRLS_loss(
                [real_item.unsqueeze(0)],
                [fake.unsqueeze(0)],
            )
            if not torch.isnan(item):
                adversarial.append(item)
            feature_matching.append(
                sum(
                    F.l1_loss(real_map[index, :length], fake_map[index, :length])
                    for real_map, fake_map in zip(real_features, fake_features, strict=True)
                )
            )
        if adversarial:
            adversarial_loss = torch.stack(adversarial).mean()
        else:
            adversarial_loss = fake_scores.sum() * 0
        return adversarial_loss, torch.stack(feature_matching).mean()


class ProsodyDiscriminatorLoss(nn.Module):
    """Discriminator objective from the author script."""

    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        predicted: Tensor,
        real: Tensor,
        style: Tensor,
        lengths: Tensor,
        max_size: int,
    ) -> Tensor:
        fake_scores, _ = self.discriminator(
            predicted.detach(),
            style.detach(),
            lengths,
            max_size,
        )
        real_scores, _ = self.discriminator(
            real.detach(),
            style.detach(),
            lengths,
            max_size,
        )
        losses = []
        for index, length_value in enumerate(lengths):
            length = int(length_value.item())
            fake = fake_scores[index, :length]
            real_item = real_scores[index, :length]
            item = (
                torch.mean(fake.square())
                + torch.mean((1 - real_item).square())
                + discriminator_TPRLS_loss(
                    [real_item.unsqueeze(0)],
                    [fake.unsqueeze(0)],
                )
            )
            if not torch.isnan(item):
                losses.append(item)
        if losses:
            return torch.stack(losses).mean()
        # The author's TPRLS subset can be empty for every item in a batch.
        # Keep the skipped discriminator update differentiable under DDP.
        return (fake_scores.sum() + real_scores.sum()) * 0
