import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def supervised_contrastive_loss(
    embeddings: Tensor,
    group_ids: Tensor,
    temperature: float,
    positive_weights: Tensor | None = None,
) -> Tensor:
    same_group = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    not_self = ~torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )
    positives = same_group & not_self
    torch._assert_async(
        torch.all(positives.sum(dim=1) > 0),
        "every contrastive embedding requires a positive partner",
    )
    if positive_weights is None:
        weights = positives.to(dtype=embeddings.dtype)
    else:
        torch._assert_async(
            torch.all(positive_weights >= 0),
            "positive weights must be a nonnegative [B,B] tensor",
        )
        weights = torch.where(
            positives,
            positive_weights.to(dtype=embeddings.dtype),
            torch.zeros_like(positive_weights, dtype=embeddings.dtype),
        )
        torch._assert_async(
            torch.all(weights.sum(dim=1) > 0),
            "positive weights must select a partner for every embedding",
        )
    normalized = F.normalize(embeddings, dim=1)
    logits = normalized @ normalized.transpose(0, 1) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_logits = logits.masked_fill(~not_self, -torch.inf)
    log_probability = logits - torch.logsumexp(
        denominator_logits,
        dim=1,
        keepdim=True,
    )
    per_embedding = -(weights * log_probability).sum(dim=1) / weights.sum(dim=1)
    return per_embedding.mean()


class GE2ELoss(nn.Module):
    def __init__(self, initial_scale: float, initial_bias: float) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        self.bias = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, embeddings: Tensor, group_ids: Tensor) -> Tensor:
        groups, inverse, counts = torch.unique(
            group_ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        torch._assert_async(
            torch.all(counts >= 2),
            "GE2E requires at least two groups with two views each",
        )
        normalized = F.normalize(embeddings, dim=1)
        group_sums = embeddings.new_zeros(groups.shape[0], embeddings.shape[1])
        group_sums.index_add_(0, inverse, normalized)
        centroids = F.normalize(group_sums / counts.unsqueeze(1), dim=1)
        own_centroid = F.normalize(
            (group_sums[inverse] - normalized) / (counts[inverse] - 1).unsqueeze(1),
            dim=1,
        )
        similarities = normalized @ centroids.transpose(0, 1)
        own_similarity = (normalized * own_centroid).sum(dim=1)
        own_group = F.one_hot(inverse, num_classes=groups.shape[0]).to(
            dtype=embeddings.dtype
        )
        similarities = similarities * (1 - own_group)
        similarities = similarities + own_similarity.unsqueeze(1) * own_group
        logits = torch.exp(self.log_scale) * similarities + self.bias
        return F.cross_entropy(logits, inverse)


def speaker_adversarial_loss(logits: Tensor, speaker_ids: Tensor) -> Tensor:
    return F.cross_entropy(logits, speaker_ids)


def style_statistics_loss(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    return F.mse_loss(predicted, target)


def embedding_consistency_loss(
    generated: Tensor,
    reference: Tensor,
    cosine_weight: float,
    mse_weight: float,
) -> Tensor:
    cosine = (1 - F.cosine_similarity(generated, reference, dim=1)).mean()
    mse = F.mse_loss(generated, reference)
    return cosine * cosine_weight + mse * mse_weight
