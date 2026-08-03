import logging

import torch
from accelerate import Accelerator
from torch import Tensor, nn


logger = logging.getLogger(__name__)


@torch.no_grad()
def initialize_rvq_codebooks(
    quantizer: nn.Module,
    continuous_style: Tensor,
    accelerator: Accelerator,
) -> None:
    """Bootstrap the author's unchanged RVQ when its pretrained checkpoint is absent."""
    if bool(quantizer._codebooks_initialized.item()):
        return
    residual = accelerator.gather(continuous_style.detach()).float()
    initialized = []
    for layer in quantizer.quantizers:
        encoded = layer.in_proj(residual)
        vectors = encoded.transpose(1, 2).reshape(-1, encoded.size(1))
        codebook_size = layer.codebook.num_embeddings
        source_indices = torch.arange(codebook_size, device=vectors.device)
        source_indices = source_indices.remainder(vectors.size(0))
        layer.codebook.weight.copy_(vectors[source_indices])
        quantized, _, _, _ = layer(residual)
        residual = residual - quantized
        initialized.append(min(vectors.size(0), codebook_size))
    quantizer._codebooks_initialized.fill_(True)
    logger.info("bootstrapped StyleTTS-ZS RVQ codebooks vectors=%s", initialized)
